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
from genesis.engine.materials.MPM.base import MPMModel, MPMPlasticity
from genesis.engine.materials.MPM.elastic import (
    func_cofactor,
    func_compression_penalty,
    func_diagonal,
    func_diagonal_matrix,
    func_fixed_corotated_principal,
    func_isotropic_pair_block,
    func_isotropic_projected_modes,
    func_product,
    func_stable_neohookean_principal,
)
from genesis.engine.materials.MPM.elasto_plastic import func_clamp_return_map, func_von_mises_return_map
from genesis.engine.materials.MPM.liquid import func_liquid_principal, func_liquid_return_map
from genesis.engine.materials.MPM.muscle import func_muscle_fibre_terms
from genesis.engine.materials.MPM.sand import func_extended_hencky_principal, func_sand_return_map
from genesis.engine.materials.MPM.snow import func_snow_hardening, func_snow_return_map
from genesis.engine.numerics.bcoo import (
    BCOOMatrixScratch,
    bcoo_matrix_sizes,
    func_add_by_key,
    func_add_triplet,
    func_read_entry,
    func_report_triplet_demand,
    func_write_triplet,
)
from genesis.engine.numerics.exclusive_sum import ExclusiveSumScratch, exclusive_sum_sizes, func_exclusive_sum
from genesis.engine.numerics.hash_table import (
    HashTableScratch,
    func_clear_hash,
    func_insert_hash,
    func_lookup_hash,
    hash_table_sizes,
)
from genesis.engine.numerics.radix_sort import RadixSortScratch, func_radix_sort, kernel_radix_sort, radix_sort_sizes
from genesis.engine.numerics.soft_contact import (
    PYRAMID_ROWS,
    func_band_row,
    func_contact_band_width,
    func_contact_friction,
    func_contact_invweight,
    func_contact_sol_params,
    func_pyramid_coefficient,
    func_pyramid_rows,
)
from genesis.engine.solvers.rigid.abd.forward_dynamics import func_vel_at_point
from genesis.engine.solvers.rigid.abd.misc import func_apply_coupling_force
from genesis.engine.solvers.rigid.data import GEOMS, LINKS, GeomsInfo, GeomsState, LinksInfo, LinksState, RigidInfo
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
    iter_state_arrays,
    opt_in,
)
from genesis.utils.geom import (
    PARTICLES,
    SpatialHashGrid,
    func_sort_particles_by_slot,
    hash_grid_sizes,
)
from genesis.utils.misc import (
    broadcast_tensor,
    qd_destroy_data,
    qd_isfinite,
    qd_to_numpy,
    qd_to_torch,
    qd_zero_grad,
)
from genesis.utils.sdf import SDFInfo
from genesis.utils.simt import qd_block_max, qd_block_shuffle, qd_block_sum, qd_segmented_sum, qd_slot_neighbors

from .base_solver import ErrorCode, SimulatorState, kernel_set_actu
from .particle_solver import ParticleSolver

if TYPE_CHECKING:
    from genesis.engine.scene import Scene
    from genesis.engine.simulator import Simulator


class MPMErrorCode(ErrorCode):
    """The error flags the material point method (MPM) solver writes in its errno (see 'Solver.register_errors')."""

    INVALID_VELOCITY_NAN = auto()
    OVERFLOW_GRID_RANGE = auto()


# The cells of the grid of the material point method (MPM) solver along each axis
GRID_X = Size("grid_x")
GRID_Y = Size("grid_y")
GRID_Z = Size("grid_z")
# The tiles of cells of the grid, and the groups of environments a warp gathers the cells of a tile for (see MPMTiles)
TILES = Size("tiles")
ENVS_GROUPS = Size("envs_groups")
# The bins of the exact accumulation of the coupling forces of the links (see MPMCouplingForces)
COUPLING_FORCE_BINS = Size("coupling_force_bins")
# The arrays of the sparse reset of the grid, which the backward pass and the implicit solver leave out (see MPMGrid)
SPARSE_GRID_RESET = "sparse_grid_reset"
# The arrays of the compatible particle-in-cell (CPIC) separation of the particles from the cells by the rigid geoms
CPIC = "cpic"
# The arrays of the deterministic transfer of the particles to the grid (see MPMConfig)
DETERMINISTIC = "deterministic"
# The arrays the backward pass alone reads
GRAD = "grad"
# The arrays of the explicit transfers, which the implicit solver leaves out, and those of the implicit solver, which
# the explicit one leaves out
EXPLICIT = "explicit"
IMPLICIT = "implicit"
# The arrays of the contact of the particles of the implicit solver with the rigid geoms
RIGID_CONTACT = "rigid_contact"

# The distinct materials of the solver, and the particles of every environment, sorted by grid cell under the implicit
# solver (see MPMSortedParticles)
MATERIALS = Size("materials")
SORTED_PARTICLES = Size("sorted_particles")
# The blocks of the sparse grid of the implicit solver, the particle blocks among them, and their nodes (see
# MPMSparseGrid)
BLOCKS = Size("blocks")
PARTICLE_BLOCKS = Size("particle_blocks")
# The cells of a block of the sparse grid along each axis, and in all
BLOCK_DIM = 4
BLOCK_CELLS = BLOCK_DIM**3
# The block coordinates a key holds, 14 bits per axis biased to be positive, and the environments, in the bits above
# them (see 'func_block_key')
BLOCK_COORD_BIAS = 8192
MAX_KEY_ENVS = 1 << 16
# The smallest capacity of the sparse grid in blocks, and the slack of a capacity grown past an overflow
MIN_BLOCKS = 64
BLOCKS_GROWTH_FACTOR = 1.2


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
    F_tmp: qd.Tensor = field((float, (3, 3)), kind=DataKind.SCRATCH, labels=EXPLICIT, grad=True)
    U: qd.Tensor = field((float, (3, 3)), kind=DataKind.SCRATCH, labels=EXPLICIT, grad=True)
    V: qd.Tensor = field((float, (3, 3)), kind=DataKind.SCRATCH, labels=EXPLICIT, grad=True)
    S: qd.Tensor = field((float, (3, 3)), kind=DataKind.SCRATCH, labels=EXPLICIT, grad=True)
    # The actuation of the muscles
    actu: qd.Tensor = field(float, grad=True)
    # The plastic volume ratio
    Jp: qd.Tensor = field(float, grad=True)
    is_active: qd.Tensor = field(bool)
    # The momentum per unit offset a particle gives the cells of its stencil, its stress and affine velocity together
    affine: qd.Tensor = field((float, (3, 3)), kind=DataKind.SCRATCH, grad=True, labels=DETERMINISTIC)
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
    # The inverse weight of a particle in its contact rows under the implicit solver, in the frame of the material at
    # rest (see 'func_contact_invweight' in genesis/engine/numerics/soft_contact.py)
    contact_invweight: qd.Tensor = field((float, (3, 3)), kind=DataKind.CONSTANT, labels=IMPLICIT)


class MPMGrid(Data, kind=DataKind.SCRATCH, shape=(GRID_X, GRID_Y, GRID_Z, ENVS)):
    """
    The background grid of the material point method (MPM), one value per cell and environment.

    `vel_in` holds the momentum the particles scatter, and `vel_out` the velocity of the cells once the grid is updated.
    Without gradient, the cells touched by a substep are zeroed right after it, listed once across the environments:
    `dirty_flag` marks a listed cell, and `dirty_list` holds the flat indices of the `dirty_count` listed cells. With
    gradient, the grid is zeroed whole with its adjoint at the start of every substep instead.
    """

    mass: qd.Tensor = field(float, labels=EXPLICIT, grad=True)
    vel_in: qd.Tensor = field((float, 3), labels=EXPLICIT, grad=True)
    vel_out: qd.Tensor = field((float, 3), labels=EXPLICIT, grad=True)
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


class MPMSortedParticles(Data, kind=DataKind.SCRATCH, shape=(SORTED_PARTICLES,)):
    """
    The particles of every environment sorted by the cell of the sparse grid they lie in, under the implicit solver.

    A particle is keyed by its environment, its block and its cell in the block (see 'func_particle_key'), and
    `particles_idx` holds the particle of every sorted slot, flattened as `i_b * n_particles + i_p`, the inactive ones
    last. The first `n_valid` slots are active. `cells_idx` numbers the cell of every slot among the cells, whose first
    slots `cells_start` lists. The slots of a cell are split into tasks of a warp at most on GPU, `tasks_idx` numbering
    the task of every slot, the assembly of the Hessian taking one per task. The terms of the Newton solve that act on
    the displacement of a particle are staged at its slot (see genesis/engine/solvers/newton_solver.py).
    """

    keys: qd.Tensor = field(qd.u64, ndarray=True)
    keys_buffer: qd.Tensor = field(qd.u64, ndarray=True)
    particles_idx: qd.Tensor = field(int, ndarray=True)
    particles_idx_buffer: qd.Tensor = field(int, ndarray=True)
    sorter: RadixSortScratch
    is_cell_start: qd.Tensor = field(int, ndarray=True)
    cells_idx: qd.Tensor = field(int, ndarray=True)
    cells_scanner: ExclusiveSumScratch
    cells_start: qd.Tensor = field(int, shape=(SORTED_PARTICLES + 1,), ndarray=True)
    is_task_start: qd.Tensor = field(int, ndarray=True)
    tasks_idx: qd.Tensor = field(int, ndarray=True)
    tasks_scanner: ExclusiveSumScratch
    n_keys: qd.Tensor = field(int, shape=(), ndarray=True)
    n_valid: qd.Tensor = field(int, shape=(), ndarray=True)
    n_cells: qd.Tensor = field(int, shape=(), ndarray=True)
    n_tasks: qd.Tensor = field(int, shape=(), ndarray=True)
    # The factor of the elastic gradient of a particle at every node of its stencil, and the gradient of the terms on
    # its displacement (see 'func_particle_stencil_terms')
    stress_factor: qd.Tensor = field((float, (3, 3)), ndarray=True)
    point_gradient: qd.Tensor = field((float, 3), ndarray=True)
    # The spring pulling a constrained particle toward its target, its stiffness dt^2 k and damping dt c, zero for a
    # particle left free, the position of the target at the end of the substep and its displacement over the substep
    constraint_spring: qd.Tensor = field(float, ndarray=True)
    constraint_damping: qd.Tensor = field(float, ndarray=True)
    constraint_target: qd.Tensor = field((float, 3), ndarray=True)
    constraint_target_disp: qd.Tensor = field((float, 3), ndarray=True)
    # The rows of the walls of the domain along each axis, the inward normal of the nearer wall of an axis being
    # `walls_sign` times the axis, and a zero stiffness standing for no row
    walls_stiffness: qd.Tensor = field((float, 3), ndarray=True)
    walls_target: qd.Tensor = field((float, 3), ndarray=True)
    walls_sign: qd.Tensor = field((float, 3), ndarray=True)


class MPMRigidContacts(Data, kind=DataKind.SCRATCH, shape=(SORTED_PARTICLES,)):
    """
    The contact of every sorted particle with the nearest rigid geom that couples with the particles, under the implicit
    solver, predicted from the state at the start of the substep (see 'kernel_mpm_predict').

    `geom_idx` is the geom, -1 for none, which leaves the other arrays unset. `stiffness` and `target` are the rows of
    the friction pyramid of the contact (see 'func_pyramid_rows'), of outward normal `normal` and friction `friction`.
    `gradient` and `hessian` are those of its rows in the displacement of the particle relative to the point of the geom
    it touches, at the current iterate of the Newton solve, which the MPM solver and the rigid solver map onto their own
    unknowns (see 'kernel_assemble_mpm_rigid_coupling' in newton_solver.py).
    """

    geom_idx: qd.Tensor = field(int, ndarray=True)
    normal: qd.Tensor = field((float, 3), ndarray=True)
    friction: qd.Tensor = field(float, ndarray=True)
    stiffness: qd.Tensor = field((float, PYRAMID_ROWS), ndarray=True)
    target: qd.Tensor = field((float, PYRAMID_ROWS), ndarray=True)
    gradient: qd.Tensor = field((float, 3), ndarray=True)
    hessian: qd.Tensor = field((float, (3, 3)), ndarray=True)


class MPMTiles(Data, kind=DataKind.DERIVED, shape=(TILES * ENVS_GROUPS,)):
    """The cubic tiles of cells of the grid that the stencils of the particles cover, by group of environments."""

    # Whether the lowest cell of the stencil of some particle of the group lies in the tile
    is_occupied: qd.Tensor = field(bool, shape=(TILES, ENVS_GROUPS))
    # The tiles some stencil of the group covers, `active_tile` being the flat index of each of the `n_active` tiles of
    # group `active_envs_group`, in no particular order
    n_active: qd.Tensor = field(int, shape=(1,))
    active_tile: qd.Tensor = field(int)
    active_envs_group: qd.Tensor = field(int)


class MPMCouplingForces(Data, kind=DataKind.SCRATCH, shape=(LINKS, COUPLING_FORCE_BINS, 6, ENVS)):
    """The forces and torques the MPM particles and grid apply the rigid links, summed exactly whatever their order."""

    # The low and high 32 bits of the 64-bit integer sum of each bin, the terms of a force or torque component being cut
    # exactly into multiples of the unit 2^(e + b k) of bin k, e and b being 'coupling_force_bin_exp_min' and
    # 'coupling_force_bin_bits' of MPMInfo (see func_add_coupling_force). With e = -120, b = 40 and 6 bins, a term below
    # 2^120 in magnitude adds at most 2^40 to a bin, so a bin holds 2^23 terms per link and substep before it overflows.
    bins_lo: qd.Tensor = field(int)
    bins_hi: qd.Tensor = field(int)


class MPMState(Data, kind=DataKind.STATE):
    """Runtime state of the material point method (MPM) solver."""

    particles: MPMParticles
    grid: MPMGrid
    particles_cpic: MPMParticlesCPIC = field(labels=CPIC)
    # The particles sorted by the lowest cell of their stencil, a slot of the hash grid hashing each cell
    hash_grid: SpatialHashGrid = field(labels=DETERMINISTIC)
    tiles: MPMTiles = field(labels=DETERMINISTIC)
    coupling_forces: MPMCouplingForces = field(labels=DETERMINISTIC)
    sorted_particles: MPMSortedParticles = field(labels=IMPLICIT)
    rigid_contacts: MPMRigidContacts = field(labels=RIGID_CONTACT)


class MPMMaterialsInfo(Data, kind=DataKind.CONSTANT, shape=(MATERIALS,)):
    """
    The parameters of the distinct materials of the solver under the implicit solver, one value per material.

    `model` is the energy density of a material (see 'MPMModel') and `plasticity` its return map (see 'MPMPlasticity'),
    `mu` and `lam` its Lame parameters, `fibre_stiffness` the active stiffness of a muscle (zero for any other
    material), `yield_lower`, `yield_higher` and `yield_stress` the yield of an elasto-plastic material and `sand_alpha`
    the friction coefficient of sand.
    """

    model: qd.Tensor = field(int)
    plasticity: qd.Tensor = field(int)
    mu: qd.Tensor = field(float)
    lam: qd.Tensor = field(float)
    fibre_stiffness: qd.Tensor = field(float)
    yield_lower: qd.Tensor = field(float)
    yield_higher: qd.Tensor = field(float)
    yield_stress: qd.Tensor = field(float)
    sand_alpha: qd.Tensor = field(float)


class MPMSparseGrid(Data, kind=DataKind.SCRATCH):
    """
    The blocks of the sparse grid of the implicit solver, rebuilt every substep from the particles, with their nodes.

    The blocks are those of the cells of the particles, the particle blocks, and the blocks of the positive octant
    around them, which hold the nodes of their cells (the neighbor blocks). `blocks_key` lists them in increasing key,
    the hash table mapping a key to its block. A node is numbered `i_blk * BLOCK_CELLS + i_cell_local` (see
    'func_cell_local'), the cell of its lower corner holding the sorted particles from `nodes_cell_start` to
    `nodes_cell_end`. `n_blocks_demanded` counts the blocks the particles need, `has_overflow` telling the grid to grow
    past its capacity. The node arrays hold the mass and the free velocity the particles give the nodes,
    `nodes_is_fixed` the nodes a fixed particle touches, and the displacement of the nodes over the substep, the unknown
    of the Newton solve, with the start point and the direction of its line search.
    """

    n_particle_blocks: qd.Tensor = field(int, shape=(), ndarray=True)
    n_neighbor_keys: qd.Tensor = field(int, shape=(), ndarray=True)
    n_blocks: qd.Tensor = field(int, shape=(), ndarray=True)
    n_blocks_demanded: qd.Tensor = field(int, shape=(), ndarray=True)
    has_overflow: qd.Tensor = field(int, shape=(), ndarray=True)
    particle_blocks_key: qd.Tensor = field(qd.u64, shape=(PARTICLE_BLOCKS,), ndarray=True)
    neighbors_key: qd.Tensor = field(qd.u64, shape=(8 * PARTICLE_BLOCKS,), ndarray=True)
    neighbors_key_buffer: qd.Tensor = field(qd.u64, shape=(8 * PARTICLE_BLOCKS,), ndarray=True)
    neighbors_source_idx: qd.Tensor = field(int, shape=(8 * PARTICLE_BLOCKS,), ndarray=True)
    neighbors_source_idx_buffer: qd.Tensor = field(int, shape=(8 * PARTICLE_BLOCKS,), ndarray=True)
    neighbors_sorter: RadixSortScratch
    neighbors_is_block_start: qd.Tensor = field(int, shape=(8 * PARTICLE_BLOCKS,), ndarray=True)
    neighbors_block_idx: qd.Tensor = field(int, shape=(8 * PARTICLE_BLOCKS,), ndarray=True)
    neighbors_scanner: ExclusiveSumScratch
    blocks_key: qd.Tensor = field(qd.u64, shape=(BLOCKS,), ndarray=True)
    hash_table: HashTableScratch
    nodes_cell_start: qd.Tensor = field(int, shape=(BLOCKS * BLOCK_CELLS,), ndarray=True)
    nodes_cell_end: qd.Tensor = field(int, shape=(BLOCKS * BLOCK_CELLS,), ndarray=True)
    nodes_mass: qd.Tensor = field(float, shape=(BLOCKS * BLOCK_CELLS,), ndarray=True)
    nodes_is_fixed: qd.Tensor = field(int, shape=(BLOCKS * BLOCK_CELLS,), ndarray=True)
    nodes_vel_free: qd.Tensor = field((float, 3), shape=(BLOCKS * BLOCK_CELLS,), ndarray=True)
    nodes_u: qd.Tensor = field((float, 3), shape=(BLOCKS * BLOCK_CELLS,), ndarray=True)
    nodes_u_start: qd.Tensor = field((float, 3), shape=(BLOCKS * BLOCK_CELLS,), ndarray=True)
    nodes_du: qd.Tensor = field((float, 3), shape=(BLOCKS * BLOCK_CELLS,), ndarray=True)
    # The diagonal block of the Hessian of every node at rest, from which the inverse weights of the particles in their
    # contact rows are computed at build (see 'MPMParticlesInfo.contact_invweight')
    nodes_diagonal: qd.Tensor = field((float, (3, 3)), shape=(BLOCKS * BLOCK_CELLS,), ndarray=True)


def build_grid(
    mpm_state: "MPMState",
    sparse_grid: MPMSparseGrid | None,
    mpm_info: "MPMInfo",
    errno: qd.Tensor,
    sparse_grids_replaced: list[MPMSparseGrid],
    n_blocks_min: int = MIN_BLOCKS,
) -> MPMSparseGrid:
    """
    Build the grid of the implicit solver over the active particles, sorted by cell, and return it.

    The grid is replaced by a larger one, of `n_blocks_min` blocks at least, whenever the particles need more blocks
    than it holds. A grid replaced keeps referenced in `sparse_grids_replaced`, with its arrays freed.
    """
    kernel_sort_particles(mpm_state, mpm_info, errno)
    n_blocks = n_blocks_min
    while True:
        if sparse_grid is None or sparse_grid.blocks_key.shape[0] < n_blocks:
            # FIXME: quadrants#1074 - a launch is cached under the identity of its struct arguments, which a new struct
            # could take over from a dead one, so the grids replaced stay referenced, with their arrays freed.
            if sparse_grid is not None:
                for _, array in iter_state_arrays(sparse_grid):
                    qd_destroy_data(array)
                sparse_grids_replaced.append(sparse_grid)
            sparse_grid = MPMSparseGrid(
                sizes={
                    BLOCKS: n_blocks,
                    PARTICLE_BLOCKS: n_blocks,
                    MPMSparseGrid.neighbors_sorter: radix_sort_sizes(key_bits=64, capacity=8 * n_blocks),
                    MPMSparseGrid.neighbors_scanner: exclusive_sum_sizes(8 * n_blocks),
                    MPMSparseGrid.hash_table: hash_table_sizes(n_blocks),
                }
            )
        kernel_list_blocks(mpm_state, sparse_grid)
        kernel_radix_sort(
            sparse_grid.neighbors_sorter,
            sparse_grid.neighbors_key,
            sparse_grid.neighbors_key_buffer,
            sparse_grid.neighbors_source_idx,
            sparse_grid.neighbors_source_idx_buffer,
            sparse_grid.n_neighbor_keys,
            key_bits=64,
        )
        kernel_publish_blocks(sparse_grid)
        if not qd_to_numpy(sparse_grid.has_overflow):
            break
        n_blocks_demanded = max(
            int(qd_to_numpy(sparse_grid.n_blocks_demanded)), int(qd_to_numpy(sparse_grid.n_particle_blocks))
        )
        n_blocks = max(int(np.ceil(n_blocks_demanded * BLOCKS_GROWTH_FACTOR)), sparse_grid.blocks_key.shape[0] + 1)
    kernel_list_cells(mpm_state, sparse_grid)
    return sparse_grid


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
    # The exponent of the unit of the lowest bin of the exact sums of the coupling forces, and the bits between the
    # units of two bins (see MPMCouplingForces)
    coupling_force_bin_exp_min: qd.Tensor = field(int, shape=(), labels=DETERMINISTIC)
    coupling_force_bin_bits: qd.Tensor = field(int, shape=(), labels=DETERMINISTIC)
    # The parameters of the distinct materials under the implicit solver
    materials: MPMMaterialsInfo = field(labels=IMPLICIT)
    # The compression over which the stiffness of a liquid fades in (see 'func_liquid_volume_terms'), and the smallest
    # volume ratio a liquid keeps
    liquid_band: qd.Tensor = field(float, shape=(), labels=IMPLICIT)
    liquid_J_min: qd.Tensor = field(float, shape=(), labels=IMPLICIT)
    # The singular value below which the extended Hencky energy continues its log linearly
    hencky_sigma_min: qd.Tensor = field(float, shape=(), labels=IMPLICIT)
    # The compression penalty of the elastic energies (see 'func_compression_penalty')
    extension_sigma_min: qd.Tensor = field(float, shape=(), labels=IMPLICIT)
    extension_stiffness: qd.Tensor = field(float, shape=(), labels=IMPLICIT)
    # The fraction of the step to the first zero volume of a liquid particle along a Newton direction that the line
    # search starts from at most
    inversion_step_slack: qd.Tensor = field(float, shape=(), labels=IMPLICIT)
    # The solver parameters of the walls of the domain, in the form of those of the rigid solver
    wall_sol_params: qd.Tensor = field((float, 7), shape=(), labels=IMPLICIT)
    # The smallest friction coefficient of the contact of a particle with a rigid geom (see 'func_contact_friction')
    contact_min_friction: qd.Tensor = field(float, shape=(), labels=IMPLICIT)


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
    # Whether the transfer to the grid and the coupling forces sum in a fixed order on GPU, with
    # `use_deterministic_algorithms` (see kernel_transfer_particles_to_grid and MPMCouplingForces)
    is_deterministic: bool
    # Whether the particles move by the implicit solver (see 'MPMOptions.use_implicit_solver')
    use_implicit_solver: bool
    # Whether the particles of the implicit solver touch the rigid geoms that couple with them (see
    # 'LegacyCouplerOptions.rigid_mpm')
    has_rigid_contact: bool
    # Whether the rigid links hold their info per environment (see 'RigidOptions.batch_links_info')
    is_links_info_batched: bool


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
        is_implicit = solver.mpm_config.use_implicit_solver
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
                coupling_force_bin_exp_min=-120,
                coupling_force_bin_bits=40,
            )
            if is_implicit:
                wall_sol_params = gu.default_solver_params()
                wall_sol_params[0] = max(solver._wall_timeconst, 2.0 * solver.substep_dt)
                info_values.update(
                    liquid_band=solver._liquid_band,
                    liquid_J_min=0.2,
                    hencky_sigma_min=0.05,
                    extension_sigma_min=0.4,
                    extension_stiffness=20.0,
                    inversion_step_slack=0.9,
                    wall_sol_params=tuple(wall_sol_params.tolist()),
                    contact_min_friction=1e-2,
                )
            hash_grid_values = dict(cell_size=solver._dx, grid_res=(grid_x, grid_y, grid_z))
            values = dict(mpm_info=info_values, mpm_state=dict(hash_grid=hash_grid_values))
            unused = {
                SPARSE_GRID_RESET: solver._requires_grad or is_implicit,
                CPIC: not solver.mpm_config.is_cpic,
                DETERMINISTIC: not solver.mpm_config.is_deterministic,
                GRAD: not solver._requires_grad,
                EXPLICIT: is_implicit,
                IMPLICIT: not is_implicit,
                RIGID_CONTACT: not solver.mpm_config.has_rigid_contact,
            }
        # The sorted particles span every environment, padded to whole warps for the sums by environment over them
        n_sorted_particles = max((solver._B * solver._n_particles + 255) // 256 * 256, 256)
        super().__init__(
            sizes={
                "particles": solver._n_particles,
                "envs": solver._B,
                "grid_x": grid_x,
                "grid_y": grid_y,
                "grid_z": grid_z,
                "geoms": solver._sim.rigid_solver.n_geoms_,
                # The slots of the hash grid hash the cells of the grid, one per particle
                **hash_grid_sizes(max(solver._n_particles, 1)),
                "tiles": n_tiles,
                "envs_groups": (solver._B + 31) // 32,
                "links": solver._sim.rigid_solver.n_links_,
                "coupling_force_bins": 6,
                "materials": max(len(solver._materials), 1),
                "sorted_particles": n_sorted_particles,
                MPMData.mpm_state.sorted_particles.sorter: radix_sort_sizes(key_bits=64, capacity=n_sorted_particles),
                MPMData.mpm_state.sorted_particles.cells_scanner: exclusive_sum_sizes(n_sorted_particles),
                MPMData.mpm_state.sorted_particles.tasks_scanner: exclusive_sum_sizes(n_sorted_particles),
            },
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
                F_new, S_new, Jp_new = self.materials[j_mat].update_F_S_Jp(J, F_tmp, U, S, V, Jp)
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
        self._use_implicit_solver = options.use_implicit_solver
        self._wall_timeconst = options.wall_timeconst
        self._liquid_smoothing_velocity = options.liquid_smoothing_velocity
        # The rigid geoms separate the particles from the cells by their SDF (see MPMParticlesCPIC)
        opt_in(SDFInfo, cond=lambda: self.is_active and self._enable_CPIC)

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
        # The implicit solver allocates the blocks of the grid its particles reach alone (see MPMSparseGrid)
        if not self._use_implicit_solver and np.prod(self._grid_res) > 1e9:
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
        # The blocks of the grid of the implicit solver, rebuilt with a larger capacity when the particles outgrow it,
        # and the grids it replaced, kept referenced with their arrays freed (see 'build_grid')
        self.sparse_grid: MPMSparseGrid | None = None
        self._sparse_grids_replaced: list[MPMSparseGrid] = []

    def build(self) -> None:
        super().build()

        if self.is_active and self._enable_CPIC and self._requires_grad:
            gs.raise_exception(
                "CPIC is not supported in differentiable mode yet. Submit a feature request if you need it."
            )

        is_implicit = self.is_active and self._use_implicit_solver
        if is_implicit:
            if self._B > MAX_KEY_ENVS:
                gs.raise_exception(f"The implicit MPM solver supports {MAX_KEY_ENVS} environments at most.")
            # The cells of a particle and of the blocks around it must fit in the keys of the sparse grid
            cells = np.concatenate((self._lower_bound, self._upper_bound)) * self._grid_density
            if cells.min() < -4 * BLOCK_COORD_BIAS or cells.max() >= 4 * (BLOCK_COORD_BIAS - 1):
                gs.raise_exception(
                    f"The MPM domain spans the cells [{cells.min():.0f}, {cells.max():.0f}], beyond the range "
                    f"[{-4 * BLOCK_COORD_BIAS}, {4 * (BLOCK_COORD_BIAS - 1)}) of the sparse grid of the implicit "
                    "solver."
                )
            liquid_smoothing_velocity = self._liquid_smoothing_velocity
            if liquid_smoothing_velocity is None:
                liquid_smoothing_velocity = self._scene.options.newton.velocity_tol
            self._liquid_band = liquid_smoothing_velocity * self.substep_dt / self._dx
            if self._liquid_band >= 0.5:
                gs.raise_exception(
                    f"The smoothing band of the liquid energy, {self._liquid_band:.3g} in volume ratio, would soften "
                    "the liquid down to half its volume: lower MPMOptions(liquid_smoothing_velocity), "
                    "NewtonOptions(velocity_tol) or the substep."
                )

        # When no material reads U, S, V, the transfer to the grid takes J from the determinant of the advected
        # deformation gradient, which spares the singular value decomposition. The backward pass reverses it alone.
        # TODO: validate the J = det(F_tmp) shortcut composes with quadrants autograd before enabling it there.
        self.mpm_config = MPMConfig(
            is_svd_needed=self._requires_grad or any([material.needs_svd for material in self._materials]),
            requires_grad=self._requires_grad,
            is_cpic=self._enable_CPIC and self._sim.rigid_solver.is_active,
            # The threads of the CPU sum each environment in a fixed order already, and the implicit solver has no
            # transfer to the grid of its own
            is_deterministic=gs.use_deterministic_algorithms and gs.backend != gs.cpu and not is_implicit,
            use_implicit_solver=self._use_implicit_solver,
            has_rigid_contact=is_implicit and self._sim.coupler.rigid_couplings[0],
            is_links_info_batched=self._sim.rigid_solver.rigid_config.batch_links_info,
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
                MPMErrorCode.INVALID_VELOCITY_NAN: (
                    """Invalid MPM particle velocities causing 'nan' or 'inf'.
Please decrease the simulation timestep or adjust the material parameters."""
                ),
                MPMErrorCode.OVERFLOW_GRID_RANGE: (
                    "MPM particles have positions that are not finite, or left the range the sparse grid of the "
                    "implicit solver indexes."
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

            if self._use_implicit_solver:
                # The energy density and the return map of every material (see 'MPMModel' and 'MPMPlasticity')
                materials_model, materials_plasticity = [], []
                for material in self._materials:
                    if not isinstance(material, (MPM.Elastic, MPM.ElastoPlastic, MPM.Liquid, MPM.Sand)):
                        gs.raise_exception(
                            "The implicit MPM solver supports the Elastic, ElastoPlastic, Liquid, Muscle, Sand and "
                            f"Snow materials alone, got {type(material).__name__}: set "
                            "MPMOptions(use_implicit_solver=False)."
                        )
                    model = MPMModel.FIXED_COROTATED
                    plasticity = MPMPlasticity.NONE
                    if isinstance(material, MPM.Sand):
                        model, plasticity = MPMModel.EXTENDED_HENCKY, MPMPlasticity.SAND
                    elif isinstance(material, MPM.Liquid):
                        model, plasticity = MPMModel.LIQUID, MPMPlasticity.LIQUID
                    elif isinstance(material, MPM.Snow):
                        plasticity = MPMPlasticity.SNOW
                    elif isinstance(material, MPM.ElastoPlastic):
                        plasticity = MPMPlasticity.VON_MISES if material.use_von_mises else MPMPlasticity.CLAMP
                    elif isinstance(material, MPM.Elastic) and material.model == "neohooken":
                        model = MPMModel.STABLE_NEOHOOKEAN
                    materials_model.append(model)
                    materials_plasticity.append(plasticity)
                materials_info = self.mpm_info.materials
                materials_values = [
                    (materials_info.model, materials_model),
                    (materials_info.plasticity, materials_plasticity),
                    (materials_info.mu, [material.mu for material in self._materials]),
                    (materials_info.lam, [material.lam for material in self._materials]),
                    (
                        materials_info.fibre_stiffness,
                        [
                            material.stiffness if isinstance(material, MPM.Muscle) else 0.0
                            for material in self._materials
                        ],
                    ),
                    (
                        materials_info.yield_lower,
                        [
                            material.yield_lower if isinstance(material, MPM.ElastoPlastic) else 0.0
                            for material in self._materials
                        ],
                    ),
                    (
                        materials_info.yield_higher,
                        [
                            material.yield_higher if isinstance(material, MPM.ElastoPlastic) else 0.0
                            for material in self._materials
                        ],
                    ),
                    (
                        materials_info.yield_stress,
                        [
                            material.von_mises_yield_stress if isinstance(material, MPM.ElastoPlastic) else 0.0
                            for material in self._materials
                        ],
                    ),
                    (
                        materials_info.sand_alpha,
                        [material.alpha if isinstance(material, MPM.Sand) else 0.0 for material in self._materials],
                    ),
                ]
                for array, values in materials_values:
                    array.from_numpy(np.array(values, dtype=gs.np_float if array.dtype == gs.qd_float else gs.np_int))

                # The first grid covers the blocks of the particles where they are added, with some slack, and grows
                # once they outgrow it (see 'build_grid')
                particles_pos = qd_to_numpy(self.mpm_state.particles.pos, transpose=True)
                cells = np.floor(particles_pos * self._inv_dx).astype(gs.np_int)
                envs = np.broadcast_to(np.arange(self._B)[:, None, None], (*cells.shape[:2], 1))
                particle_blocks = np.unique(np.concatenate((envs, cells // BLOCK_DIM), axis=-1).reshape(-1, 4), axis=0)
                octant = np.array([[0, (a >> 2) & 1, (a >> 1) & 1, a & 1] for a in range(8)])
                n_blocks = len(np.unique((particle_blocks[:, None] + octant).reshape(-1, 4), axis=0))
                self.sparse_grid = build_grid(
                    mpm_state=self.mpm_state,
                    sparse_grid=None,
                    mpm_info=self.mpm_info,
                    errno=self._errno,
                    sparse_grids_replaced=self._sparse_grids_replaced,
                    n_blocks_min=2 * n_blocks,
                )
                kernel_mpm_p2g(
                    mpm_state=self.mpm_state,
                    sparse_grid=self.sparse_grid,
                    sim_state=self._sim.sim_state,
                    mpm_info=self.mpm_info,
                    force_fields=self.force_fields,
                )
                kernel_contact_invweight(mpm_state=self.mpm_state, sparse_grid=self.sparse_grid, mpm_info=self.mpm_info)
            else:
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
            self.mpm_config,
        )

    # ------------------------------------------------------------------------------------
    # ------------------------------------ stepping --------------------------------------
    # ------------------------------------------------------------------------------------

    def substep_pre_coupling(self) -> None:
        if self._use_implicit_solver:
            # The Newton solve of the substep moves the nodes (see NewtonSolver), from the grid laid over the particles
            self.sparse_grid = build_grid(
                self.mpm_state, self.sparse_grid, self.mpm_info, self._errno, self._sparse_grids_replaced
            )
            kernel_mpm_p2g(self.mpm_state, self.sparse_grid, self._sim.sim_state, self.mpm_info, self.force_fields)
            rigid_solver = self._sim.rigid_solver
            kernel_mpm_predict(
                self.mpm_state,
                rigid_solver.dyn_state.links,
                rigid_solver.dyn_state.geoms,
                self.particle_constraints,
                self.mpm_info,
                rigid_solver.dyn_info.links,
                rigid_solver.dyn_info.geoms,
                rigid_solver.rigid_info,
                rigid_solver.collider.sdf.sdf_info,
                self.mpm_config,
                rigid_solver.collider.collider_config,
                self._is_constrained,
                rigid_solver.is_solved_in_global_newton,
            )
            return
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
        if self._use_implicit_solver:
            kernel_mpm_g2p(self.mpm_state, self.sparse_grid, self.mpm_info, self._errno, self._sim.newton_solver._errno)
            kernel_copy_next_to_curr(self.mpm_state, self._errno)
            return
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
        return not self._enable_CPIC and not self._use_implicit_solver

    @property
    def data(self) -> Iterator[DataItem | ConfigItem]:
        yield from iter_data(self, "mpm_config")
        yield from iter_data(self.data_manager, "mpm_info")
        yield from iter_data(self, "particle_constraints")
        yield from iter_data(self.data_manager, "mpm_state")
        yield from iter_data(self.data_manager, "errno", DataKind.STATE)
        if self.sparse_grid is not None:
            yield from iter_data(self, "sparse_grid")

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
    # The bins of the coupling forces start the substep empty, including in the replay of a step by the backward pass,
    # which restores them to the values a later substep left
    if qd.static(mpm_config.is_deterministic and not is_backward):
        for i_l, i_k, i_c, i_b in qd.ndrange(*mpm_state.coupling_forces.bins_lo.shape):
            mpm_state.coupling_forces.bins_lo[i_l, i_k, i_c, i_b] = 0
            mpm_state.coupling_forces.bins_hi[i_l, i_k, i_c, i_b] = 0

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
            if qd.static(mpm_config.is_deterministic):
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
                                        mpm_state.grid.dirty_list[i_slot] = (
                                            cell[0] * grid_y + cell[1]
                                        ) * grid_z + cell[2]
                        # The particles held in place act as boundary conditions of the grid. The zero races with the
                        # scatter of the other particles, which the variant autodiff reverses defers past it instead.
                        if qd.static(not is_backward):
                            if not mpm_info.particles.is_free[i_p]:
                                cell = base - grid_offset + offset
                                mpm_state.grid.vel_in[cell[0], cell[1], cell[2], i_b] = qd.Vector.zero(gs.qd_float, 3)

    if qd.static(mpm_config.is_deterministic and not is_backward):
        func_sort_particles(mpm_state, mpm_info, mpm_config)
        func_gather_particles(mpm_state, mpm_info, mpm_config, is_g2p_backward=False)
    elif qd.static(is_backward):
        # The particles held in place zero the cells of their stencils once every particle scattered into them
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
                            mpm_state.grid.dirty_list[i_slot] = (cell[0] * grid_res[1] + cell[1]) * grid_res[2] + cell[
                                2
                            ]


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
def func_add_coupling_force(
    i_l: int,
    i_b: int,
    pos: qd.types.vector(3),
    force: qd.types.vector(3),
    links_state: LinksState,
    coupling_forces: MPMCouplingForces,
    mpm_info: MPMInfo,
    is_exact: qd.template(),
):
    """
    Apply a force at a point of a rigid link, its reaction on the MPM particles or grid.

    With `is_exact`, the force and its torque add exactly to integer bins (see MPMCouplingForces), whatever the order
    of the terms, which kernel_flush_coupling_forces adds to the link. The cut of a term into bins is exact: from the
    highest bin down, the integer part of the term at the scale of a bin holds the leading bits of its mantissa, and the
    remainder the trailing ones. A bin adds its 64 bits as two 32-bit halves, the low half carrying into the high one.
    """
    if qd.static(is_exact):
        torque = (pos - links_state.root_COM[i_l, i_b]).cross(force)
        # A runtime loop over the components keeps a single inlined copy of the bin updates
        for i_c in range(6):
            wrench = qd.select(i_c < 3, torque, force)
            i_d = i_c % 3
            remainder = -qd.select(i_d == 0, wrench[0], qd.select(i_d == 1, wrench[1], wrench[2]))
            for k_ in range(coupling_forces.bins_lo.shape[1]):
                k = coupling_forces.bins_lo.shape[1] - 1 - k_
                scale = qd.pow(
                    2.0,
                    qd.cast(
                        mpm_info.coupling_force_bin_exp_min[None] + mpm_info.coupling_force_bin_bits[None] * k,
                        gs.qd_float,
                    ),
                )
                count = qd.cast(remainder / scale, qd.i64)
                if count != 0:
                    remainder = remainder - qd.cast(count, gs.qd_float) * scale
                    count_lo = qd.cast(count & qd.i64(0xFFFFFFFF), qd.u32)
                    lo_prev = qd.cast(
                        qd.atomic_add(coupling_forces.bins_lo[i_l, k, i_c, i_b], qd.cast(count_lo, gs.qd_int)), qd.u32
                    )
                    carry = qd.select(lo_prev + count_lo < lo_prev, 1, 0)
                    qd.atomic_add(coupling_forces.bins_hi[i_l, k, i_c, i_b], qd.cast(count >> 32, gs.qd_int) + carry)
    else:
        func_apply_coupling_force(i_l, i_b, pos, force, links_state)


@no_grad
@qd.kernel(fastcache=True)
def kernel_flush_coupling_forces(links_state: LinksState, coupling_forces: MPMCouplingForces, mpm_info: MPMInfo):
    """Add to the coupling forces of the links the exact sums of their bins, from the highest bin down."""
    for i_l, i_b in qd.ndrange(coupling_forces.bins_lo.shape[0], coupling_forces.bins_lo.shape[3]):
        terms = qd.Vector.zero(gs.qd_float, 6)
        for i_c in qd.static(range(6)):
            for k_ in range(coupling_forces.bins_lo.shape[1]):
                k = coupling_forces.bins_lo.shape[1] - 1 - k_
                count = (qd.cast(coupling_forces.bins_hi[i_l, k, i_c, i_b], qd.i64) << 32) | qd.cast(
                    qd.cast(coupling_forces.bins_lo[i_l, k, i_c, i_b], qd.u32), qd.i64
                )
                scale = qd.pow(
                    2.0,
                    qd.cast(
                        mpm_info.coupling_force_bin_exp_min[None] + mpm_info.coupling_force_bin_bits[None] * k,
                        gs.qd_float,
                    ),
                )
                terms[i_c] = terms[i_c] + qd.cast(count, gs.qd_float) * scale
        links_state.cfrc_coupling_ang[i_l, i_b] = links_state.cfrc_coupling_ang[i_l, i_b] + qd.Vector(
            [terms[0], terms[1], terms[2]]
        )
        links_state.cfrc_coupling_vel[i_l, i_b] = links_state.cfrc_coupling_vel[i_l, i_b] + qd.Vector(
            [terms[3], terms[4], terms[5]]
        )


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
    of the collision of the particle with the geom instead, which pushes the link of the geom (see
    'func_point_vel_after_two_way_collision_with_rigid_geom').
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
                    func_add_coupling_force(
                        geoms_info.link_idx[i_g_sep],
                        i_b,
                        pos,
                        force,
                        links_state,
                        mpm_state.coupling_forces,
                        mpm_info,
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


@qd.kernel
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


@qd.kernel
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
@qd.kernel
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
    mpm_config: qd.template(),
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
        if qd.static(mpm_config.use_implicit_solver):
            # The inverse mass of the particle alone, which a particle inactive when the inverse weights of the contact
            # rows are computed keeps (see 'kernel_contact_invweight')
            volume_real = mpm_info.particle_volume[None] / mpm_info.particle_volume_scale[None]
            mpm_info.particles.contact_invweight[i_p] = qd.Matrix.identity(gs.qd_float, 3) / (volume_real * rho)

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


# ------------------------------------------------------------------------------------
# --------------------- implicit solver: sparse grid and stencils --------------------
# ------------------------------------------------------------------------------------

# The implicit solver moves the particles by the displacement over the substep of the nodes of the grid around them, the
# trilinear (Q1) interpolation over the 8 nodes of the cell of a particle, the node of the lower corner of a cell
# standing for the cell. The nodes live in blocks of 4 x 4 x 4 cells, which the sparse grid allocates where the
# particles reach (see MPMSparseGrid).


@qd.func
def func_node_offset(a: int):
    """Return the offset of a node of a cell from its lower corner, the 8 nodes of the cell counted in binary."""
    return qd.Vector([(a >> 2) & 1, (a >> 1) & 1, a & 1], dt=gs.qd_int)


@qd.func
def func_cell_local(cell):
    """Return the index of a cell or of a node in its block."""
    return ((cell[0] & 3) << 4) | ((cell[1] & 3) << 2) | (cell[2] & 3)


@qd.func
def func_local_cell(i_local: int):
    """Return the offset in its block of the cell of an index (see 'func_cell_local')."""
    return qd.Vector([(i_local >> 4) & 3, (i_local >> 2) & 3, i_local & 3], dt=gs.qd_int)


@qd.func
def func_block_key(i_b: int, block):
    """Return the key of a block of an environment: the environment from bit 42 on, then 14 bits per block coordinate,
    biased to be positive, so that the keys of an environment sort by block coordinates."""
    return (
        (qd.u64(i_b) << qd.u64(42))
        | (qd.u64(block[0] + 8192) << qd.u64(28))
        | (qd.u64(block[1] + 8192) << qd.u64(14))
        | qd.u64(block[2] + 8192)
    )


@qd.func
def func_block_of_key(key: qd.u64):
    """Return the environment and the block coordinates of a block key (see 'func_block_key')."""
    block = qd.Vector(
        [
            gs.qd_int((key >> qd.u64(28)) & qd.u64(0x3FFF)) - 8192,
            gs.qd_int((key >> qd.u64(14)) & qd.u64(0x3FFF)) - 8192,
            gs.qd_int(key & qd.u64(0x3FFF)) - 8192,
        ],
        dt=gs.qd_int,
    )
    return gs.qd_int(key >> qd.u64(42)), block


@qd.func
def func_q1_node_weight(offset, fx: qd.types.vector(3), inv_dx: float):
    """
    Return the trilinear weight of a node of a cell at a point of the cell, and its gradient in the position of the
    point.

    The node lies at `offset` from the lower corner of the cell, and the point `fx` cells from that corner.
    """
    offset_float = offset.cast(gs.qd_float)
    axes_w = offset_float * fx + (1.0 - offset_float) * (1.0 - fx)
    axes_dw = (2.0 * offset_float - 1.0) * inv_dx
    return func_product(axes_w), axes_dw * func_cofactor(axes_w)


@qd.func
def func_q1_stencil(fx: qd.types.vector(3), inv_dx: float):
    """
    Return the trilinear weights of the 8 nodes of a cell at a point of the cell, and their gradients in the position of
    the point, node a in row a.

    The point lies `fx` cells from the lower corner of the cell.
    """
    w = qd.Vector.zero(gs.qd_float, 8)
    grad_w = qd.Matrix.zero(gs.qd_float, 8, 3)
    for a in qd.static(range(8)):
        w_a, grad_w_a = func_q1_node_weight(func_node_offset(a), fx, inv_dx)
        w[a] = w_a
        for d in qd.static(range(3)):
            grad_w[a, d] = grad_w_a[d]
    return w, grad_w


@qd.func
def func_sorted_particle(i_s: int, n_particles: int, sorted_particles: MPMSortedParticles):
    """Return the particle, the environment and the cell of a sorted slot."""
    i_flat = sorted_particles.particles_idx[i_s]
    i_b = i_flat // n_particles
    key = sorted_particles.keys[i_s]
    _, block = func_block_of_key(key >> qd.u64(6))
    return i_flat - i_b * n_particles, i_b, block * 4 + func_local_cell(gs.qd_int(key & qd.u64(63)))


@qd.func
def func_stencil_slots(i_b: int, cell, sparse_grid: MPMSparseGrid):
    """Return the slots of the 8 nodes of a cell, all of which the sparse grid holds."""
    slots = qd.Vector.zero(gs.qd_int, 8)
    for a in qd.static(range(8)):
        node = cell + func_node_offset(a)
        i_blk = func_lookup_hash(sparse_grid.hash_table, func_block_key(i_b, node >> 2))
        slots[a] = i_blk * 64 + func_cell_local(node)
    return slots


@qd.func
def func_gather_stencil(slots, nodes_values: qd.Tensor):
    """Return the vectors of the 8 nodes of a stencil, node a in row a."""
    values = qd.Matrix.zero(gs.qd_float, 8, 3)
    for a in qd.static(range(8)):
        node_value = nodes_values[slots[a]]
        for d in qd.static(range(3)):
            values[a, d] = node_value[d]
    return values


@qd.func
def func_cell_particles(i_b: int, cell, i_lane: int, sparse_grid: MPMSparseGrid, is_lanes: qd.template()):
    """Return the range of the sorted slots of the particles of a cell, looked up once and shared by the lanes of a warp
    when the lanes gather a node together."""
    i_s_start = gs.qd_int(0)
    i_s_end = gs.qd_int(0)
    if i_lane == 0:
        i_blk = func_lookup_hash(sparse_grid.hash_table, func_block_key(i_b, cell >> 2))
        if i_blk >= 0:
            i_n = i_blk * 64 + func_cell_local(cell)
            i_s_start = sparse_grid.nodes_cell_start[i_n]
            i_s_end = sparse_grid.nodes_cell_end[i_n]
    if qd.static(is_lanes):
        i_s_start = qd_block_shuffle(i_s_start, 0)
        i_s_end = qd_block_shuffle(i_s_end, 0)
    return i_s_start, i_s_end


@qd.kernel(graph=True)
def kernel_sort_particles(mpm_state: MPMState, mpm_info: MPMInfo, errno: qd.Tensor):
    """
    Key every particle of every environment by the cell it lies in, and sort them by key.

    The inactive particles take the largest key, which sorts last. A particle out of the range of the keys flags its
    environment and takes it too.
    """
    n_particles, n_envs = mpm_state.particles.pos.shape
    for i_s in range(mpm_state.sorted_particles.keys.shape[0]):
        key = qd.u64(0xFFFFFFFFFFFFFFFF)
        if i_s < n_envs * n_particles:
            i_b = i_s // n_particles
            i_p = i_s - i_b * n_particles
            if mpm_state.particles.is_active[i_p, i_b]:
                pos = mpm_state.particles.pos[i_p, i_b]
                x = pos * mpm_info.inv_dx[None]
                # Fast math keeps the test of the position as loaded, where it may fold away a test of a computed result
                is_in_range = qd_isfinite(pos) and ((x >= -32768.0) & (x < 32764.0)).all()
                if is_in_range:
                    cell = qd.floor(x).cast(gs.qd_int)
                    key = (func_block_key(i_b, cell >> 2) << qd.u64(6)) | qd.u64(func_cell_local(cell))
                else:
                    qd.atomic_or(errno[i_b], MPMErrorCode.OVERFLOW_GRID_RANGE)
        mpm_state.sorted_particles.keys[i_s] = key
        mpm_state.sorted_particles.particles_idx[i_s] = i_s
    for _ in range(1):
        mpm_state.sorted_particles.n_keys[None] = mpm_state.sorted_particles.keys.shape[0]
    func_radix_sort(
        mpm_state.sorted_particles.sorter,
        mpm_state.sorted_particles.keys,
        mpm_state.sorted_particles.keys_buffer,
        mpm_state.sorted_particles.particles_idx,
        mpm_state.sorted_particles.particles_idx_buffer,
        mpm_state.sorted_particles.n_keys,
        key_bits=64,
    )


@qd.kernel
def kernel_list_blocks(mpm_state: MPMState, sparse_grid: MPMSparseGrid):
    """Count the active sorted particles, list the distinct blocks of their cells, and the 8 blocks of the positive
    octant of each, which hold the nodes of its cells."""
    n_slots = mpm_state.sorted_particles.keys.shape[0]
    for _ in range(1):
        mpm_state.sorted_particles.n_valid[None] = 0
        sparse_grid.n_particle_blocks[None] = 0
        sparse_grid.has_overflow[None] = 0
    for i_s in range(n_slots):
        key = mpm_state.sorted_particles.keys[i_s]
        if key != qd.u64(0xFFFFFFFFFFFFFFFF):
            is_last_valid = i_s == n_slots - 1
            if not is_last_valid:
                is_last_valid = mpm_state.sorted_particles.keys[i_s + 1] == qd.u64(0xFFFFFFFFFFFFFFFF)
            if is_last_valid:
                mpm_state.sorted_particles.n_valid[None] = i_s + 1
            is_block_start = i_s == 0
            if not is_block_start:
                is_block_start = (mpm_state.sorted_particles.keys[i_s - 1] >> qd.u64(6)) != (key >> qd.u64(6))
            if is_block_start:
                i_blk = qd.atomic_add(sparse_grid.n_particle_blocks[None], 1)
                if i_blk < sparse_grid.particle_blocks_key.shape[0]:
                    sparse_grid.particle_blocks_key[i_blk] = key >> qd.u64(6)
    for _ in range(1):
        n_particle_blocks = sparse_grid.n_particle_blocks[None]
        if n_particle_blocks > sparse_grid.particle_blocks_key.shape[0]:
            n_particle_blocks = sparse_grid.particle_blocks_key.shape[0]
            sparse_grid.has_overflow[None] = 1
        sparse_grid.n_neighbor_keys[None] = 8 * n_particle_blocks
    for i_key in range(sparse_grid.n_neighbor_keys[None]):
        i_b, block = func_block_of_key(sparse_grid.particle_blocks_key[i_key >> 3])
        sparse_grid.neighbors_key[i_key] = func_block_key(i_b, block + func_node_offset(i_key & 7))
        sparse_grid.neighbors_source_idx[i_key] = i_key


@qd.kernel
def kernel_publish_blocks(sparse_grid: MPMSparseGrid):
    """Number the distinct blocks of the sorted neighbor keys in increasing key and insert them in the hash table, the
    blocks past the capacity flagging the grid to grow."""
    func_clear_hash(sparse_grid.hash_table)
    for i_key in range(sparse_grid.n_neighbor_keys[None]):
        is_block_start = 1
        if i_key > 0:
            if sparse_grid.neighbors_key[i_key] == sparse_grid.neighbors_key[i_key - 1]:
                is_block_start = 0
        sparse_grid.neighbors_is_block_start[i_key] = is_block_start
    func_exclusive_sum(
        sparse_grid.neighbors_scanner,
        sparse_grid.neighbors_is_block_start,
        sparse_grid.neighbors_block_idx,
        sparse_grid.n_neighbor_keys,
    )
    for i_key in range(sparse_grid.n_neighbor_keys[None]):
        if sparse_grid.neighbors_is_block_start[i_key] != 0:
            i_blk = sparse_grid.neighbors_block_idx[i_key]
            if i_blk < sparse_grid.blocks_key.shape[0]:
                sparse_grid.blocks_key[i_blk] = sparse_grid.neighbors_key[i_key]
                func_insert_hash(sparse_grid.hash_table, sparse_grid.neighbors_key[i_key], i_blk)
    for _ in range(1):
        n_keys = sparse_grid.n_neighbor_keys[None]
        n_blocks_demanded = gs.qd_int(0)
        if n_keys > 0:
            n_blocks_demanded = (
                sparse_grid.neighbors_block_idx[n_keys - 1] + sparse_grid.neighbors_is_block_start[n_keys - 1]
            )
        sparse_grid.n_blocks_demanded[None] = n_blocks_demanded
        sparse_grid.n_blocks[None] = qd.min(n_blocks_demanded, sparse_grid.blocks_key.shape[0])
        if n_blocks_demanded > sparse_grid.blocks_key.shape[0]:
            sparse_grid.has_overflow[None] = 1


@qd.kernel
def kernel_list_cells(mpm_state: MPMState, sparse_grid: MPMSparseGrid):
    """
    Number the cells of the active sorted particles and their tasks, and give every node the sorted particles of the
    cell of its lower corner.

    A task is a run of particles of one cell, cut at every warp on GPU, which one warp assembles together.
    """
    for i_s in range(mpm_state.sorted_particles.n_valid[None]):
        is_cell_start = 1
        if i_s > 0:
            if mpm_state.sorted_particles.keys[i_s] == mpm_state.sorted_particles.keys[i_s - 1]:
                is_cell_start = 0
        is_task_start = is_cell_start
        if qd.static(gs.backend != gs.cpu):
            if i_s % 32 == 0:
                is_task_start = 1
        mpm_state.sorted_particles.is_cell_start[i_s] = is_cell_start
        mpm_state.sorted_particles.is_task_start[i_s] = is_task_start
    func_exclusive_sum(
        mpm_state.sorted_particles.cells_scanner,
        mpm_state.sorted_particles.is_cell_start,
        mpm_state.sorted_particles.cells_idx,
        mpm_state.sorted_particles.n_valid,
    )
    func_exclusive_sum(
        mpm_state.sorted_particles.tasks_scanner,
        mpm_state.sorted_particles.is_task_start,
        mpm_state.sorted_particles.tasks_idx,
        mpm_state.sorted_particles.n_valid,
    )
    for _ in range(1):
        mpm_state.sorted_particles.n_cells[None] = 0
        mpm_state.sorted_particles.n_tasks[None] = 0
    for i_s in range(mpm_state.sorted_particles.n_valid[None]):
        # The exclusive sums count the starts before a slot, the start of the slot itself opening its cell and its task
        i_cell = mpm_state.sorted_particles.cells_idx[i_s] + mpm_state.sorted_particles.is_cell_start[i_s] - 1
        i_task = mpm_state.sorted_particles.tasks_idx[i_s] + mpm_state.sorted_particles.is_task_start[i_s] - 1
        mpm_state.sorted_particles.cells_idx[i_s] = i_cell
        mpm_state.sorted_particles.tasks_idx[i_s] = i_task
        if mpm_state.sorted_particles.is_cell_start[i_s] != 0:
            mpm_state.sorted_particles.cells_start[i_cell] = i_s
        if i_s == mpm_state.sorted_particles.n_valid[None] - 1:
            mpm_state.sorted_particles.n_cells[None] = i_cell + 1
            mpm_state.sorted_particles.n_tasks[None] = i_task + 1
    for _ in range(1):
        n_cells = mpm_state.sorted_particles.n_cells[None]
        mpm_state.sorted_particles.cells_start[n_cells] = mpm_state.sorted_particles.n_valid[None]
    for i_n in range(sparse_grid.n_blocks[None] * 64):
        sparse_grid.nodes_cell_start[i_n] = 0
        sparse_grid.nodes_cell_end[i_n] = 0
    for i_cell in range(mpm_state.sorted_particles.n_cells[None]):
        i_s_start = mpm_state.sorted_particles.cells_start[i_cell]
        key = mpm_state.sorted_particles.keys[i_s_start]
        i_blk = func_lookup_hash(sparse_grid.hash_table, key >> qd.u64(6))
        if i_blk >= 0:
            i_n = i_blk * 64 + gs.qd_int(key & qd.u64(63))
            sparse_grid.nodes_cell_start[i_n] = i_s_start
            sparse_grid.nodes_cell_end[i_n] = mpm_state.sorted_particles.cells_start[i_cell + 1]


@qd.kernel
def kernel_mpm_p2g(
    mpm_state: MPMState,
    sparse_grid: MPMSparseGrid,
    sim_state: SimulatorState,
    mpm_info: MPMInfo,
    force_fields: qd.template(),
):
    """
    Transfer the mass and the momentum of the particles to the nodes of the sparse grid by affine particle-in-cell
    (APIC), and set the free velocity of every node over the substep, under gravity and the force fields.

    Every node gathers the particles of the 8 cells it is a node of, the lanes of a warp sharing a node on GPU. A node
    that a particle held in place touches is fixed. The displacement of every node starts at zero.
    """
    N_LANES = qd.static(1 if gs.backend == gs.cpu else 32)
    n_particles = mpm_state.particles.pos.shape[0]
    inv_dx = mpm_info.inv_dx[None]
    dt = mpm_info.substep_dt[None]
    qd.loop_config(block_dim=128)
    for i_th in range(sparse_grid.n_blocks[None] * 64 * N_LANES):
        i_lane = gs.qd_int(0)
        if qd.static(N_LANES > 1):
            i_lane = gs.qd_int(qd.simt.subgroup.invocation_id()) & 31
        i_n = i_th // N_LANES
        i_b, block = func_block_of_key(sparse_grid.blocks_key[i_n // 64])
        node = block * 4 + func_local_cell(i_n % 64)
        x_n = node.cast(gs.qd_float) * mpm_info.dx[None]
        mass = gs.qd_float(0.0)
        momentum = qd.Vector.zero(gs.qd_float, 3)
        is_fixed = 0
        for a in range(8):
            offset = func_node_offset(a)
            cell = node - offset
            i_s_start, i_s_end = func_cell_particles(i_b, cell, i_lane, sparse_grid, is_lanes=N_LANES > 1)
            for i_chunk_ in range((i_s_end - i_s_start + N_LANES - 1) // N_LANES):
                i_s = i_s_start + i_chunk_ * N_LANES + i_lane
                if i_s < i_s_end:
                    i_p = mpm_state.sorted_particles.particles_idx[i_s] - i_b * n_particles
                    x_p = mpm_state.particles.pos[i_p, i_b]
                    w, _ = func_q1_node_weight(offset, x_p * inv_dx - cell.cast(gs.qd_float), inv_dx)
                    mass_p = w * mpm_info.particles.mass[i_p] / mpm_info.particle_volume_scale[None]
                    vel_p = mpm_state.particles.vel[i_p, i_b] + dt * mpm_info.gravity[i_b]
                    mass += mass_p
                    momentum += mass_p * (vel_p + mpm_state.particles.C[i_p, i_b] @ (x_n - x_p))
                    if not mpm_info.particles.is_free[i_p]:
                        is_fixed = 1
        if qd.static(N_LANES > 1):
            mass = qd_block_sum(mass)
            for d in qd.static(range(3)):
                momentum[d] = qd_block_sum(momentum[d])
            is_fixed = qd_block_max(is_fixed)
        if i_lane == 0:
            vel = qd.Vector.zero(gs.qd_float, 3)
            if mass > 0.0 and is_fixed == 0:
                vel = momentum / mass
                acc = force_fields.add_acc(
                    acc=qd.Vector.zero(gs.qd_float, 3), pos=x_n, vel=vel, t=sim_state.t[i_b], i_p=-1
                )
                vel = vel + acc * dt
            sparse_grid.nodes_mass[i_n] = mass
            sparse_grid.nodes_is_fixed[i_n] = is_fixed
            sparse_grid.nodes_vel_free[i_n] = vel
            sparse_grid.nodes_u[i_n] = qd.Vector.zero(gs.qd_float, 3)
            sparse_grid.nodes_u_start[i_n] = qd.Vector.zero(gs.qd_float, 3)
            sparse_grid.nodes_du[i_n] = qd.Vector.zero(gs.qd_float, 3)


@qd.func
def func_elastic_principal(
    model: int,
    sigma: qd.types.vector(3),
    mu: float,
    lam: float,
    hencky_sigma_min: float,
    liquid_band: float,
    extension_sigma_min: float,
    extension_stiffness: float,
):
    """
    Return the energy density of a material of a model (see 'MPMModel'), its gradient and its Hessian in the signed
    singular values.

    The elastic solids resist further compression past `extension_sigma_min`.
    """
    psi = gs.qd_float(0.0)
    g = qd.Vector.zero(gs.qd_float, 3)
    A = qd.Matrix.zero(gs.qd_float, 3, 3)
    if model == MPMModel.EXTENDED_HENCKY:
        psi, g, A = func_extended_hencky_principal(sigma, mu, lam, hencky_sigma_min)
    elif model == MPMModel.LIQUID:
        psi, g, A = func_liquid_principal(sigma, mu, lam, liquid_band)
    else:
        if model == MPMModel.STABLE_NEOHOOKEAN:
            psi, g, A = func_stable_neohookean_principal(sigma, mu, lam)
        else:
            psi, g, A = func_fixed_corotated_principal(sigma, mu, lam)
        psi_penalty, g_penalty, A_penalty = func_compression_penalty(
            sigma, mu, lam, extension_sigma_min, extension_stiffness
        )
        psi += psi_penalty
        g += g_penalty
        A += func_diagonal_matrix(A_penalty)
    return psi, g, A


@qd.func
def func_plastic_return_map(
    plasticity: int,
    F: qd.types.matrix(3, 3),
    Jp: float,
    mu: float,
    lam: float,
    yield_lower: float,
    yield_higher: float,
    yield_stress: float,
    sand_alpha: float,
    liquid_J_min: float,
):
    """Return the deformation gradient and the plastic state of a particle after the return map of its material (see
    'MPMPlasticity')."""
    F_new = F
    Jp_new = Jp
    if plasticity == MPMPlasticity.LIQUID:
        F_new = func_liquid_return_map(F, liquid_J_min)
    elif plasticity != MPMPlasticity.NONE:
        U, S, V = gu.qd_svd(F)
        sigma = func_diagonal(S)
        sigma_new = sigma
        is_yielding = True
        if plasticity == MPMPlasticity.VON_MISES:
            sigma_new, is_yielding = func_von_mises_return_map(sigma, mu, yield_stress)
        elif plasticity == MPMPlasticity.CLAMP:
            sigma_new = func_clamp_return_map(sigma, yield_lower, yield_higher)
        elif plasticity == MPMPlasticity.SNOW:
            sigma_new, Jp_new = func_snow_return_map(sigma, Jp, yield_lower, yield_higher)
        else:
            sigma_new, Jp_new = func_sand_return_map(sigma, Jp, mu, lam, sand_alpha)
        if is_yielding:
            F_new = U @ func_diagonal_matrix(sigma_new) @ V.transpose()
    return F_new, Jp_new


@qd.func
def func_spring_terms(
    spring: float,
    damping: float,
    target: qd.types.vector(3),
    target_disp: qd.types.vector(3),
    x: qd.types.vector(3),
    disp: qd.types.vector(3),
):
    """
    Return the energy and the gradient of the spring of a constrained particle over the substep, and the scale of its
    isotropic Hessian.

    The particle at x moves by `disp` over the substep. The spring pulls it toward the position of its target at the end
    of the substep, with the stiffness `spring` = dt^2 k, and damps its displacement relative to that of its target,
    with `damping` = dt c.
    """
    gap = x + disp - target
    disp_rel = disp - target_disp
    energy = 0.5 * spring * gap.norm_sqr() + 0.5 * damping * disp_rel.norm_sqr()
    return energy, spring * gap + damping * disp_rel, spring + damping


@qd.func
def func_walls_terms(
    stiffness: qd.types.vector(3), target: qd.types.vector(3), sign: qd.types.vector(3), disp: qd.types.vector(3)
):
    """
    Return the energy, the gradient and the Hessian of the rows of the walls of the domain on the displacement of a
    particle (see 'MPMSortedParticles.walls_stiffness').

    Every axis k holds a frictionless row of normal `sign[k]`, active while the displacement leaves the gap short of the
    target.
    """
    gap = sign * disp - target
    stiffness_active = stiffness * (gap < 0.0).cast(gs.qd_float)
    return 0.5 * stiffness_active.dot(gap * gap), stiffness_active * gap * sign, func_diagonal_matrix(stiffness_active)


@qd.func
def func_particle_material(i_mat: int, Jp: float, mpm_info: MPMInfo):
    """Return the model, the plasticity and the Lame parameters of a material, those of snow hardened by the plastic
    state of the particle."""
    plasticity = mpm_info.materials.plasticity[i_mat]
    mu = mpm_info.materials.mu[i_mat]
    lam = mpm_info.materials.lam[i_mat]
    if plasticity == MPMPlasticity.SNOW:
        hardening = func_snow_hardening(Jp)
        mu = mu * hardening
        lam = lam * hardening
    return mpm_info.materials.model[i_mat], plasticity, mu, lam


@qd.func
def func_particle_point_terms(
    i_s: int, x: qd.types.vector(3), disp: qd.types.vector(3), sorted_particles: MPMSortedParticles
):
    """
    Return the energy, the gradient and the Hessian of the terms acting on the displacement of a sorted particle over
    the substep.

    The particle at x moves by `disp`. The terms come from the rows staged at its slot: the spring of its constraint and
    the walls of the domain (see 'kernel_mpm_predict').
    """
    energy_spring, gradient_spring, curvature_spring = func_spring_terms(
        sorted_particles.constraint_spring[i_s],
        sorted_particles.constraint_damping[i_s],
        sorted_particles.constraint_target[i_s],
        sorted_particles.constraint_target_disp[i_s],
        x,
        disp,
    )
    energy_walls, gradient_walls, hessian_walls = func_walls_terms(
        sorted_particles.walls_stiffness[i_s],
        sorted_particles.walls_target[i_s],
        sorted_particles.walls_sign[i_s],
        disp,
    )
    energy = energy_spring + energy_walls
    gradient = gradient_spring + gradient_walls
    hessian = curvature_spring * qd.Matrix.identity(gs.qd_float, 3) + hessian_walls
    return energy, gradient, hessian


@qd.func
def func_particle_stencil_terms(
    i_s: int,
    gradient_contact: qd.types.vector(3),
    hessian_contact: qd.types.matrix(3, 3),
    mpm_state: MPMState,
    sparse_grid: MPMSparseGrid,
    mpm_info: MPMInfo,
):
    """
    Return the terms of the Hessian of the incremental potential of a sorted particle on the nodes of its stencil, at
    the displacement of the nodes, and stage the factors of its gradient at its slot.

    The particle contributes its elastic energy, the energy of the fibres of a muscle, the terms acting on its own
    displacement (see 'func_particle_point_terms'), and the gradient and the Hessian of its contact with a rigid geom in
    that displacement (see 'MPMRigidContacts.gradient'). The block of the nodes a and b is 'func_stencil_pair_block' of
    the terms returned. A fixed node takes no part.
    """
    n_particles = mpm_state.particles.pos.shape[0]
    inv_dx = mpm_info.inv_dx[None]
    dt = mpm_info.substep_dt[None]
    i_p, i_b, cell = func_sorted_particle(i_s, n_particles, mpm_state.sorted_particles)
    x_p = mpm_state.particles.pos[i_p, i_b]
    w, grad_w = func_q1_stencil(x_p * inv_dx - cell.cast(gs.qd_float), inv_dx)
    slots = func_stencil_slots(i_b, cell, sparse_grid)
    u_nodes = func_gather_stencil(slots, sparse_grid.nodes_u)
    movable = qd.Vector([gs.qd_float(sparse_grid.nodes_is_fixed[slots[a]] == 0) for a in qd.static(range(8))])

    # The elastic energy V dt^2 psi(F_trial) at the trial deformation gradient F_trial = (I + sum_a u_a grad w_a^T) F
    F = mpm_state.particles.F[i_p, i_b]
    F_trial = (qd.Matrix.identity(gs.qd_float, 3) + u_nodes.transpose() @ grad_w) @ F
    i_mat = mpm_info.particles.material_idx[i_p]
    model, plasticity, mu, lam = func_particle_material(i_mat, mpm_state.particles.Jp[i_p, i_b], mpm_info)
    U, S, V = gu.qd_svd(F_trial)
    sigma = func_diagonal(S)
    psi, g, A = func_elastic_principal(
        model,
        sigma,
        mu,
        lam,
        mpm_info.hencky_sigma_min[None],
        mpm_info.liquid_band[None],
        mpm_info.extension_sigma_min[None],
        mpm_info.extension_stiffness[None],
    )
    scale = mpm_info.particle_volume[None] / mpm_info.particle_volume_scale[None] * dt * dt
    muscle_direction = mpm_info.particles.muscle_direction[i_p]
    psi_fibre, P_fibre, fibre_curvature = func_muscle_fibre_terms(
        mpm_info.materials.fibre_stiffness[i_mat], mpm_state.particles.actu[i_p, i_b], muscle_direction, F_trial
    )
    P = U @ func_diagonal_matrix(g) @ V.transpose()
    S_block, alpha, beta = func_isotropic_projected_modes(sigma, g, A)

    # The terms on the displacement of the particle, sum_a w_a u_a
    energy_point, point_gradient, point_hessian = func_particle_point_terms(
        i_s, x_p, u_nodes.transpose() @ w, mpm_state.sorted_particles
    )
    point_gradient += gradient_contact
    point_hessian += hessian_contact
    mpm_state.sorted_particles.stress_factor[i_s] = scale * (P + P_fibre) @ F.transpose()
    mpm_state.sorted_particles.point_gradient[i_s] = point_gradient

    # The rows r_a = grad w_a^T F V of the stencil in the frame V, the weights of the rank-one fibre term, and the
    # weights of the terms on the displacement, all zero on a fixed node
    r_frame = (movable.outer_product(qd.Vector([1.0, 1.0, 1.0], dt=gs.qd_float))) * (grad_w @ F @ V)
    fibre_weights = qd.sqrt(scale * fibre_curvature) * (r_frame @ (V.transpose() @ muscle_direction))
    return slots, r_frame, U, scale * S_block, scale * alpha, scale * beta, fibre_weights, movable * w, point_hessian


@qd.func
def func_stencil_pair_block(
    a: int,
    b: int,
    r_frame: qd.types.matrix(8, 3),
    U: qd.types.matrix(3, 3),
    S_block: qd.types.matrix(3, 3),
    alpha: qd.types.vector(3),
    beta: qd.types.vector(3),
    fibre_weights: qd.types.vector(8),
    point_weights: qd.types.vector(8),
    point_hessian: qd.types.matrix(3, 3),
):
    """Return the Hessian block of the nodes a and b of the stencil of a particle, from its terms (see
    'func_particle_stencil_terms')."""
    r_a = qd.Vector([r_frame[a, d] for d in qd.static(range(3))], dt=gs.qd_float)
    r_b = qd.Vector([r_frame[b, d] for d in qd.static(range(3))], dt=gs.qd_float)
    block = U @ func_isotropic_pair_block(r_a, r_b, S_block, alpha, beta) @ U.transpose()
    block += fibre_weights[a] * fibre_weights[b] * qd.Matrix.identity(gs.qd_float, 3)
    return block + point_weights[a] * point_weights[b] * point_hessian


@qd.func
def func_particle_energy(
    i_s: int, energy_contact: float, mpm_state: MPMState, sparse_grid: MPMSparseGrid, mpm_info: MPMInfo
):
    """Return the energy of a sorted particle in the incremental potential at the displacement of the nodes, with the
    energy of its contact with a rigid geom (see 'func_particle_stencil_terms')."""
    n_particles = mpm_state.particles.pos.shape[0]
    inv_dx = mpm_info.inv_dx[None]
    dt = mpm_info.substep_dt[None]
    i_p, i_b, cell = func_sorted_particle(i_s, n_particles, mpm_state.sorted_particles)
    x_p = mpm_state.particles.pos[i_p, i_b]
    w, grad_w = func_q1_stencil(x_p * inv_dx - cell.cast(gs.qd_float), inv_dx)
    u_nodes = func_gather_stencil(func_stencil_slots(i_b, cell, sparse_grid), sparse_grid.nodes_u)
    F_trial = (qd.Matrix.identity(gs.qd_float, 3) + u_nodes.transpose() @ grad_w) @ mpm_state.particles.F[i_p, i_b]
    i_mat = mpm_info.particles.material_idx[i_p]
    model, plasticity, mu, lam = func_particle_material(i_mat, mpm_state.particles.Jp[i_p, i_b], mpm_info)
    U, S, V = gu.qd_svd(F_trial)
    psi, g, A = func_elastic_principal(
        model,
        func_diagonal(S),
        mu,
        lam,
        mpm_info.hencky_sigma_min[None],
        mpm_info.liquid_band[None],
        mpm_info.extension_sigma_min[None],
        mpm_info.extension_stiffness[None],
    )
    psi_fibre, P_fibre, fibre_curvature = func_muscle_fibre_terms(
        mpm_info.materials.fibre_stiffness[i_mat],
        mpm_state.particles.actu[i_p, i_b],
        mpm_info.particles.muscle_direction[i_p],
        F_trial,
    )
    energy_point, point_gradient, point_hessian = func_particle_point_terms(
        i_s, x_p, u_nodes.transpose() @ w, mpm_state.sorted_particles
    )
    energy_point += energy_contact
    scale = mpm_info.particle_volume[None] / mpm_info.particle_volume_scale[None] * dt * dt
    return i_b, scale * (psi + psi_fibre) + energy_point


@qd.func
def func_particle_displacement(
    i_s: int, nodes_disp: qd.Tensor, mpm_state: MPMState, sparse_grid: MPMSparseGrid, mpm_info: MPMInfo
):
    """Return the environment of a sorted particle and its displacement sum_a w_a u_a from a displacement of the nodes,
    their iterate or their update along the Newton direction."""
    n_particles = mpm_state.particles.pos.shape[0]
    inv_dx = mpm_info.inv_dx[None]
    i_p, i_b, cell = func_sorted_particle(i_s, n_particles, mpm_state.sorted_particles)
    w, _ = func_q1_stencil(mpm_state.particles.pos[i_p, i_b] * inv_dx - cell.cast(gs.qd_float), inv_dx)
    disp_nodes = func_gather_stencil(func_stencil_slots(i_b, cell, sparse_grid), nodes_disp)
    return i_b, disp_nodes.transpose() @ w


@qd.func
def func_det_cubic(F: qd.types.matrix(3, 3), dF: qd.types.matrix(3, 3)):
    """Return the coefficients of det(F + t dF) in the powers of t, from the constant one up."""
    a = [F[:, c] for c in qd.static(range(3))]
    b = [dF[:, c] for c in qd.static(range(3))]
    c0 = a[0].dot(a[1].cross(a[2]))
    c1 = b[0].dot(a[1].cross(a[2])) + a[0].dot(b[1].cross(a[2])) + a[0].dot(a[1].cross(b[2]))
    c2 = a[0].dot(b[1].cross(b[2])) + b[0].dot(a[1].cross(b[2])) + b[0].dot(b[1].cross(a[2]))
    c3 = b[0].dot(b[1].cross(b[2]))
    return c0, c1, c2, c3


@qd.func
def func_cubic_first_root(c0: float, c1: float, c2: float, c3: float, t_max: float):
    """
    Return the smallest positive root of a cubic positive at zero, up to a bound.

    The cubic has the coefficients c0 to c3, from the constant one up. Where it stays positive up to `t_max`, the root
    returned is twice `t_max`. The extrema of the cubic split [0, t_max] into monotone pieces, the first one crossing
    zero being bisected.
    """
    t_low = t_max
    t_high = t_max
    a = 3.0 * c3
    b = 2.0 * c2
    if qd.abs(a) > 0.0:
        discriminant = b * b - 4.0 * a * c1
        if discriminant > 0.0:
            root_sqrt = qd.sqrt(discriminant)
            t_1 = (-b - root_sqrt) / (2.0 * a)
            t_2 = (-b + root_sqrt) / (2.0 * a)
            if t_1 > 0.0 and t_1 < t_max:
                t_low = t_1
            if t_2 > 0.0 and t_2 < t_max:
                t_high = t_2
    elif qd.abs(b) > 0.0:
        t_extremum = -c1 / b
        if t_extremum > 0.0 and t_extremum < t_max:
            t_low = t_extremum
    if t_high < t_low:
        t_swap = t_low
        t_low = t_high
        t_high = t_swap
    breakpoints = qd.Vector([t_low, t_high, t_max], dt=gs.qd_float)
    root = 2.0 * t_max
    is_found = False
    t_left = gs.qd_float(0.0)
    for k in qd.static(range(3)):
        t_right = breakpoints[k]
        value_left = c0 + t_left * (c1 + t_left * (c2 + t_left * c3))
        value_right = c0 + t_right * (c1 + t_right * (c2 + t_right * c3))
        if not is_found and value_left > 0.0 and value_right <= 0.0:
            is_found = True
            lo = t_left
            hi = t_right
            for _ in range(64):
                mid = 0.5 * (lo + hi)
                if c0 + mid * (c1 + mid * (c2 + mid * c3)) > 0.0:
                    lo = mid
                else:
                    hi = mid
            root = 0.5 * (lo + hi)
        t_left = t_right
    return root


@qd.func
def func_particle_inversion_step_cap(i_s: int, mpm_state: MPMState, sparse_grid: MPMSparseGrid, mpm_info: MPMInfo):
    """
    Return whether the volume of a liquid sorted particle reaches zero within a bound along the Newton direction, and
    the step that the line search starts from at most.

    The bound is the step 1 / `inversion_step_slack`, and the step returned `inversion_step_slack` times the first step
    reaching zero volume. The line search stays short of zero volume, where the energy of a liquid has no value.
    """
    is_capped = False
    alpha_cap = gs.qd_float(0.0)
    n_particles = mpm_state.particles.pos.shape[0]
    i_p, i_b, cell = func_sorted_particle(i_s, n_particles, mpm_state.sorted_particles)
    i_mat = mpm_info.particles.material_idx[i_p]
    if mpm_info.materials.model[i_mat] == MPMModel.LIQUID:
        inv_dx = mpm_info.inv_dx[None]
        _, grad_w = func_q1_stencil(mpm_state.particles.pos[i_p, i_b] * inv_dx - cell.cast(gs.qd_float), inv_dx)
        slots = func_stencil_slots(i_b, cell, sparse_grid)
        F = mpm_state.particles.F[i_p, i_b]
        F_trial = (
            qd.Matrix.identity(gs.qd_float, 3) + func_gather_stencil(slots, sparse_grid.nodes_u).transpose() @ grad_w
        ) @ F
        dF = (func_gather_stencil(slots, sparse_grid.nodes_du).transpose() @ grad_w) @ F
        c0, c1, c2, c3 = func_det_cubic(F_trial, dF)
        if c0 > 0.0:
            slack = mpm_info.inversion_step_slack[None]
            root = func_cubic_first_root(c0, c1, c2, c3, 1.0 / slack)
            if root <= 1.0 / slack:
                is_capped = True
                alpha_cap = slack * root
    return is_capped, alpha_cap


@qd.kernel
def kernel_mpm_predict(
    mpm_state: MPMState,
    links_state: LinksState,
    geoms_state: GeomsState,
    particle_constraints: MPMParticleConstraints,
    mpm_info: MPMInfo,
    links_info: LinksInfo,
    geoms_info: GeomsInfo,
    rigid_info: RigidInfo,
    sdf_info: SDFInfo,
    mpm_config: qd.template(),
    collider_config: qd.template(),
    is_constrained: qd.template(),
    is_rigid_solved_in_global_newton: qd.template(),
):
    """
    Predict the constraints of every active sorted particle over the substep from the state at its start.

    The constraints hold over the Newton solve: the rows of the walls of the domain, of the contact of the particle with
    a rigid geom and of the spring of its particle constraint, at the slot of the particle. A particle gets the row of
    the nearer wall of an axis within the band of the wall (see 'func_contact_band_width'), and the rows of the friction
    pyramid of the nearest rigid geom of its env that couples with the particles within the band of the geom, its gap
    being its signed distance short of the softness of the geom. The stiffness of a row comes from the inverse weight of
    the particle along its normal, and that of the link of the geom. The target of a spring moves with the link of the
    particle at its velocity over the substep. The links stand at the start of the substep when the global Newton solve
    solves the rigid solver (`is_rigid_solved_in_global_newton`), at its end otherwise.
    """
    n_particles = mpm_state.particles.pos.shape[0]
    dt = mpm_info.substep_dt[None]
    dx = mpm_info.dx[None]
    lower = mpm_info.boundary.lower[None]
    upper = mpm_info.boundary.upper[None]
    for i_s in range(mpm_state.sorted_particles.n_valid[None]):
        i_p, i_b, cell = func_sorted_particle(i_s, n_particles, mpm_state.sorted_particles)
        x_p = mpm_state.particles.pos[i_p, i_b]
        v_p = mpm_state.particles.vel[i_p, i_b]
        # The nearer wall of every axis, its distance positive inside the domain and the inward normal sign e_k
        gaps_upper = upper - x_p
        is_upper = gaps_upper < x_p - lower
        gaps = qd.select(is_upper, gaps_upper, x_p - lower)
        signs = 1.0 - 2.0 * is_upper.cast(gs.qd_float)
        is_near = (gaps < 2.0 * dt * qd.max(-signs * v_p, 0.0) + dx).any()

        # The nearest rigid geom, kept within reach of the particle at its speed relative to the geom, which bounds its
        # approach speed and spares the normal of the geoms out of reach
        i_g_contact = gs.qd_int(-1)
        gap_contact = gs.qd_float(0.0)
        vel_geom = qd.Vector.zero(gs.qd_float, 3)
        if qd.static(mpm_config.has_rigid_contact):
            gap_contact = gs.qd_float(qd.math.inf)
            for i_g in range(geoms_info.needs_coup.shape[0]):
                i_l = geoms_info.link_idx[i_g]
                I_l = [i_l, i_b] if qd.static(mpm_config.is_links_info_batched) else i_l
                # A heterogeneous entity carries the geoms of one of its variants per env, the others standing nowhere
                # there (see 'func_update_geoms' in rigid/abd/forward_kinematics.py)
                if geoms_info.needs_coup[i_g] and links_info.geom_start[I_l] <= i_g < links_info.geom_end[I_l]:
                    gap = (
                        sdf.sdf_func_world(i_g, i_b, x_p, geoms_state, geoms_info, sdf_info)
                        - geoms_info.coup_softness[i_g]
                    )
                    if gap < gap_contact:
                        i_g_contact = i_g
                        gap_contact = gap
            if i_g_contact >= 0:
                i_l = geoms_info.link_idx[i_g_contact]
                vel_geom = func_vel_at_point(i_l, i_b, x_p, links_state)
                if gap_contact >= func_contact_band_width((v_p - vel_geom).norm(), dt, dx):
                    i_g_contact = -1
            is_near = is_near or i_g_contact >= 0
        U = qd.Matrix.identity(gs.qd_float, 3)
        V = qd.Matrix.identity(gs.qd_float, 3)
        if is_near:
            U, S, V = gu.qd_svd(mpm_state.particles.F[i_p, i_b])

        stiffness = qd.Vector.zero(gs.qd_float, 3)
        target = qd.Vector.zero(gs.qd_float, 3)
        for k in range(3):
            if gaps[k] < func_contact_band_width(-signs[k] * v_p[k], dt, dx):
                normal = signs[k] * qd.Vector([gs.qd_float(k == d) for d in qd.static(range(3))], dt=gs.qd_float)
                invweight = func_contact_invweight(U, V, mpm_info.particles.contact_invweight[i_p], normal)
                stiffness_k, target_k = func_band_row(
                    mpm_info.wall_sol_params[None], gaps[k], signs[k] * v_p[k], invweight, dt
                )
                stiffness[k] = stiffness_k
                target[k] = target_k
        mpm_state.sorted_particles.walls_stiffness[i_s] = stiffness
        mpm_state.sorted_particles.walls_target[i_s] = target
        mpm_state.sorted_particles.walls_sign[i_s] = signs

        if qd.static(mpm_config.has_rigid_contact):
            if i_g_contact >= 0:
                normal = sdf.sdf_func_normal_world(
                    i_g_contact, i_b, x_p, geoms_state, geoms_info, rigid_info, sdf_info, collider_config
                )
                if gap_contact < func_contact_band_width(-normal.dot(v_p - vel_geom), dt, dx):
                    i_l = geoms_info.link_idx[i_g_contact]
                    I_l = [i_l, i_b] if qd.static(mpm_config.is_links_info_batched) else i_l
                    friction = func_contact_friction(
                        geoms_info.friction[i_g_contact],
                        geoms_state.friction_ratio[i_g_contact, i_b],
                        mpm_info.contact_min_friction[None],
                    )
                    invweight_particle = func_contact_invweight(U, V, mpm_info.particles.contact_invweight[i_p], normal)
                    stiffness_rigid, target_rigid = func_pyramid_rows(
                        normal,
                        friction,
                        func_pyramid_coefficient(
                            friction, rigid_info.impratio[None], invweight_particle + links_info.invweight[I_l][0]
                        ),
                        func_contact_sol_params(geoms_info.sol_params[i_g_contact], dt),
                        gap_contact,
                        v_p - vel_geom,
                        vel_geom,
                        dt,
                    )
                    mpm_state.rigid_contacts.normal[i_s] = normal
                    mpm_state.rigid_contacts.friction[i_s] = friction
                    mpm_state.rigid_contacts.stiffness[i_s] = stiffness_rigid
                    mpm_state.rigid_contacts.target[i_s] = target_rigid
                else:
                    i_g_contact = -1
            mpm_state.rigid_contacts.geom_idx[i_s] = i_g_contact

        spring = gs.qd_float(0.0)
        damping = gs.qd_float(0.0)
        target_pos = qd.Vector.zero(gs.qd_float, 3)
        target_disp = qd.Vector.zero(gs.qd_float, 3)
        if qd.static(is_constrained):
            if particle_constraints.is_constrained[i_p, i_b]:
                target_pos = particle_constraints.target_pos[i_p, i_b]
                i_l = particle_constraints.link_idx[i_p, i_b]
                if i_l >= 0:
                    target_pos = gu.qd_transform_by_trans_quat(
                        particle_constraints.link_local_pos[i_p, i_b],
                        links_state.pos[i_l, i_b],
                        links_state.quat[i_l, i_b],
                    )
                    target_disp = dt * func_vel_at_point(i_l, i_b, target_pos, links_state)
                stiffness_spring = particle_constraints.stiffness[i_p, i_b]
                mass = mpm_info.particles.mass[i_p] / mpm_info.particle_volume_scale[None]
                spring = dt * dt * stiffness_spring
                damping = dt * 2.0 * qd.sqrt(stiffness_spring * mass)
        if qd.static(is_rigid_solved_in_global_newton):
            target_pos = target_pos + target_disp
        mpm_state.sorted_particles.constraint_spring[i_s] = spring
        mpm_state.sorted_particles.constraint_damping[i_s] = damping
        mpm_state.sorted_particles.constraint_target[i_s] = target_pos
        mpm_state.sorted_particles.constraint_target_disp[i_s] = target_disp


@qd.kernel
def kernel_contact_invweight(mpm_state: MPMState, sparse_grid: MPMSparseGrid, mpm_info: MPMInfo):
    """
    Compute the inverse weight of every active particle in its contact rows at rest, from the diagonal blocks of the
    Hessian of the nodes of its stencil.

    A particle is lumped onto the nodes of its stencil, each node of diagonal block D_a weighing w_a^2 D_a^-1, the
    blocks holding the kinetic and the elastic terms alone. The inverse weight is taken in the first environment, the
    frame of the material at rest being the same in all.
    """
    n_particles = mpm_state.particles.pos.shape[0]
    inv_dx = mpm_info.inv_dx[None]
    for i_n in range(sparse_grid.n_blocks[None] * 64):
        mass = sparse_grid.nodes_mass[i_n]
        if mass <= 0.0:
            mass = 1.0
        sparse_grid.nodes_diagonal[i_n] = mass * qd.Matrix.identity(gs.qd_float, 3)
    for i_s in range(mpm_state.sorted_particles.n_valid[None]):
        slots, r_frame, U, S_block, alpha, beta, fibre_weights, point_weights, point_hessian = (
            func_particle_stencil_terms(
                i_s,
                qd.Vector.zero(gs.qd_float, 3),
                qd.Matrix.zero(gs.qd_float, 3, 3),
                mpm_state,
                sparse_grid,
                mpm_info,
            )
        )
        for a in qd.static(range(8)):
            qd.atomic_add(
                sparse_grid.nodes_diagonal[slots[a]],
                func_stencil_pair_block(
                    a,
                    a,
                    r_frame,
                    U,
                    S_block,
                    alpha,
                    beta,
                    fibre_weights,
                    point_weights,
                    qd.Matrix.zero(gs.qd_float, 3, 3),
                ),
            )
    for i_s in range(mpm_state.sorted_particles.n_valid[None]):
        i_p, i_b, cell = func_sorted_particle(i_s, n_particles, mpm_state.sorted_particles)
        if i_b == 0:
            w, _ = func_q1_stencil(mpm_state.particles.pos[i_p, i_b] * inv_dx - cell.cast(gs.qd_float), inv_dx)
            slots = func_stencil_slots(i_b, cell, sparse_grid)
            invweight = qd.Matrix.zero(gs.qd_float, 3, 3)
            for a in qd.static(range(8)):
                invweight += w[a] * w[a] * sparse_grid.nodes_diagonal[slots[a]].inverse()
            mpm_info.particles.contact_invweight[i_p] = 0.5 * (invweight + invweight.transpose())


@qd.kernel
def kernel_mpm_g2p(
    mpm_state: MPMState, sparse_grid: MPMSparseGrid, mpm_info: MPMInfo, errno: qd.Tensor, newton_errno: qd.Tensor
):
    """
    Move every active particle into the next state by the displacement of the nodes of its stencil (grid-to-particle
    transfer), then apply the return map of its material (see 'kernel_copy_next_to_curr').

    The particles of an environment that the Newton solver flagged keep their state, and so do the inactive particles
    (see 'kernel_flag_failures' in newton_solver.py). A velocity that is not finite flags the environment of the
    particle.
    """
    n_particles, n_envs = mpm_state.particles.pos.shape
    inv_dx = mpm_info.inv_dx[None]
    dt = mpm_info.substep_dt[None]
    for i_s in range(mpm_state.sorted_particles.n_valid[None]):
        i_p, i_b, cell = func_sorted_particle(i_s, n_particles, mpm_state.sorted_particles)
        if newton_errno[i_b] == 0:
            x_p = mpm_state.particles.pos[i_p, i_b]
            w, grad_w = func_q1_stencil(x_p * inv_dx - cell.cast(gs.qd_float), inv_dx)
            u_nodes = func_gather_stencil(func_stencil_slots(i_b, cell, sparse_grid), sparse_grid.nodes_u)
            vel = u_nodes.transpose() @ w / dt
            C = u_nodes.transpose() @ grad_w / dt
            i_mat = mpm_info.particles.material_idx[i_p]
            F_new, Jp_new = func_plastic_return_map(
                mpm_info.materials.plasticity[i_mat],
                (qd.Matrix.identity(gs.qd_float, 3) + dt * C) @ mpm_state.particles.F[i_p, i_b],
                mpm_state.particles.Jp[i_p, i_b],
                mpm_info.materials.mu[i_mat],
                mpm_info.materials.lam[i_mat],
                mpm_info.materials.yield_lower[i_mat],
                mpm_info.materials.yield_higher[i_mat],
                mpm_info.materials.yield_stress[i_mat],
                mpm_info.materials.sand_alpha[i_mat],
                mpm_info.liquid_J_min[None],
            )
            if not qd_isfinite(vel):
                qd.atomic_or(errno[i_b], MPMErrorCode.INVALID_VELOCITY_NAN)
            mpm_state.particles.pos_next[i_p, i_b] = x_p + dt * vel
            mpm_state.particles.vel_next[i_p, i_b] = vel
            mpm_state.particles.C_next[i_p, i_b] = C
            mpm_state.particles.F_next[i_p, i_b] = F_new
            mpm_state.particles.Jp_next[i_p, i_b] = Jp_new
    for i_p, i_b in qd.ndrange(n_particles, n_envs):
        is_kept = not mpm_state.particles.is_active[i_p, i_b]
        if newton_errno[i_b] != 0:
            is_kept = True
        if is_kept:
            mpm_state.particles.pos_next[i_p, i_b] = mpm_state.particles.pos[i_p, i_b]
            mpm_state.particles.vel_next[i_p, i_b] = mpm_state.particles.vel[i_p, i_b]
            mpm_state.particles.C_next[i_p, i_b] = mpm_state.particles.C[i_p, i_b]
            mpm_state.particles.F_next[i_p, i_b] = mpm_state.particles.F[i_p, i_b]
            mpm_state.particles.Jp_next[i_p, i_b] = mpm_state.particles.Jp[i_p, i_b]


# ------------------------------------------------------------------------------------
# -------------------------------- global Newton solve -------------------------------
# ------------------------------------------------------------------------------------

# The rows of the linear system of the global Newton solve, one per node of the sparse grid of the implicit solver
ROWS = Size("rows")
# The smallest capacity of the rows, and the slack of a capacity grown past an overflow
MIN_ROWS = 64 * 64
ROWS_GROWTH_FACTOR = 1.2
# The nodes of the stencil of a particle, the corners of its cell (see 'func_q1_stencil'), and the blocks of the upper
# triangle of the Hessian of a stencil, one per pair of nodes (a, b) with a <= b: 8 x 9 / 2 = 36
STENCIL_NODES = 8
STENCIL_PAIRS = STENCIL_NODES * (STENCIL_NODES + 1) // 2


class NewtonRows(Data, kind=DataKind.SCRATCH, shape=(ROWS,)):
    """
    The linear system of the Newton iterations, one row of 3 unknowns per node of the sparse grid, and its solve.

    `gradient` is the gradient of the incremental potential at the current iterate, the right-hand side, `solution` the
    solution of the Hessian against it, and `r`, `z`, `p` and `Ap` the residual, the preconditioned residual, the search
    direction and its product with the Hessian of the preconditioned conjugate gradient (PCG). `jacobi` holds the
    inverses of the diagonal blocks of the Hessian, the block Jacobi preconditioner. `rows_env` is the environment of
    every row, the rows of an environment following one another. The matrix holds the upper triangle of the Hessian.
    """

    n_rows: qd.Tensor = field(int, shape=(), ndarray=True)
    rows_env: qd.Tensor = field(int, ndarray=True)
    gradient: qd.Tensor = field((float, 3), ndarray=True)
    solution: qd.Tensor = field((float, 3), ndarray=True)
    r: qd.Tensor = field((float, 3), ndarray=True)
    z: qd.Tensor = field((float, 3), ndarray=True)
    p: qd.Tensor = field((float, 3), ndarray=True)
    Ap: qd.Tensor = field((float, 3), ndarray=True)
    jacobi: qd.Tensor = field((float, (3, 3)), ndarray=True)
    matrix: BCOOMatrixScratch


def grow_linear_system(
    n_rows: int, n_tasks: int, newton_rows: NewtonRows | None, newton_rows_replaced: list[NewtonRows]
) -> NewtonRows:
    """
    Return the rows of the linear system of the substep, grown past the nodes of the grid and the triplets of their
    Hessian: the rows given if they hold them, larger ones otherwise.

    The Hessian holds the kinetic block of every node and the upper triangle of the 8 x 8 blocks of the stencil of every
    task of the assembly (see 'kernel_list_cells'). Rows replaced stay referenced in `newton_rows_replaced`, with their
    arrays freed.
    """
    n_triplets = n_rows + STENCIL_PAIRS * n_tasks
    if newton_rows is not None:
        if newton_rows.rows_env.shape[0] >= n_rows and newton_rows.matrix.triplets_row.shape[0] >= n_triplets:
            return newton_rows
        # FIXME: quadrants#1074 - a launch is cached under the identity of its struct arguments, which a new struct
        # could take over from a dead one, so the rows replaced stay referenced, with their arrays freed.
        for _, array in iter_state_arrays(newton_rows):
            qd_destroy_data(array)
        newton_rows_replaced.append(newton_rows)
    max_rows = max(int(np.ceil(n_rows * ROWS_GROWTH_FACTOR)), MIN_ROWS)
    max_triplets = max(int(np.ceil(n_triplets * ROWS_GROWTH_FACTOR)), MIN_ROWS)
    return NewtonRows(sizes={ROWS: max_rows, NewtonRows.matrix: bcoo_matrix_sizes(max_triplets, block_shape=(3, 3))})


@qd.func
def func_staged_contact_terms(i_s: int, mpm_state: MPMState, mpm_config: qd.template()):
    """Return the gradient and the Hessian of the contact of a sorted particle with a rigid geom staged at the current
    iterate (see 'kernel_assemble_mpm_rigid_coupling' in newton_solver.py), zero for a particle in no contact."""
    gradient = qd.Vector.zero(gs.qd_float, 3)
    hessian = qd.Matrix.zero(gs.qd_float, 3, 3)
    if qd.static(mpm_config.has_rigid_contact):
        if mpm_state.rigid_contacts.geom_idx[i_s] >= 0:
            gradient = mpm_state.rigid_contacts.gradient[i_s]
            hessian = mpm_state.rigid_contacts.hessian[i_s]
    return gradient, hessian


@qd.func
def func_node_kinetic_energy(
    mass: float, is_fixed: int, u: qd.types.vector(3), vel_free: qd.types.vector(3), dt: float
):
    """Return the kinetic energy of a node in the incremental potential, 1/2 m |u - dt v_free|^2, zero for a fixed node
    or a node without mass."""
    energy = gs.qd_float(0.0)
    if mass > 0.0 and is_fixed == 0:
        energy = 0.5 * mass * (u - dt * vel_free).norm_sqr()
    return energy


@qd.func(requires_top_level=True)
def func_mpm_assemble(
    newton_rows: NewtonRows,
    mpm_state: MPMState,
    sparse_grid: MPMSparseGrid,
    mpm_info: MPMInfo,
    mpm_config: qd.template(),
):
    """
    Assemble the gradient and the projected Hessian of the incremental potential of the MPM solver at the displacement
    of the nodes.

    Every node takes its kinetic block, an identity on a fixed node, whose unknown stays zero, and on a node without
    mass, whose unknown the elastic terms of the particles around it set. Every task of the sorted particles (see
    'kernel_list_cells') writes the blocks of the upper triangle of the stencil of its cell (see STENCIL_PAIRS), summed
    over its particles: the lanes of a warp sum them on GPU, and one thread walks the particles of a cell on CPU. The
    gradient of a node then gathers the factors its particles staged. A particle in contact with a rigid geom adds the
    terms of its contact staged at the current iterate (see 'func_staged_contact_terms').
    """
    n_nodes = sparse_grid.n_blocks[None] * 64
    n_valid = mpm_state.sorted_particles.n_valid[None]
    for _ in range(1):
        newton_rows.n_rows[None] = n_nodes
        func_report_triplet_demand(
            newton_rows.matrix, n_nodes + STENCIL_PAIRS * mpm_state.sorted_particles.n_tasks[None]
        )
    for i_n in range(n_nodes):
        i_b, block = func_block_of_key(sparse_grid.blocks_key[i_n // 64])
        newton_rows.rows_env[i_n] = i_b
        mass = sparse_grid.nodes_mass[i_n]
        if mass <= 0.0 or sparse_grid.nodes_is_fixed[i_n] != 0:
            mass = 1.0
        func_write_triplet(
            newton_rows.matrix, i_n, i_n, i_n, mass * qd.Matrix.identity(gs.qd_float, 3), is_symmetric=True
        )

    # The pairs of nodes (a, b) of the upper triangle of a stencil, a <= b, in the order of the triplets of a task
    pairs_a = qd.Vector([a for a in range(STENCIL_NODES) for b in range(a, STENCIL_NODES)], dt=gs.qd_int)
    pairs_b = qd.Vector([b for a in range(STENCIL_NODES) for b in range(a, STENCIL_NODES)], dt=gs.qd_int)
    if qd.static(gs.backend == gs.cpu):
        for i_cell in range(mpm_state.sorted_particles.n_cells[None]):
            i_s_start = mpm_state.sorted_particles.cells_start[i_cell]
            i_t_start = n_nodes + STENCIL_PAIRS * mpm_state.sorted_particles.tasks_idx[i_s_start]
            for i_s in range(i_s_start, mpm_state.sorted_particles.cells_start[i_cell + 1]):
                gradient_contact, hessian_contact = func_staged_contact_terms(i_s, mpm_state, mpm_config)
                slots, r_frame, U, S_block, alpha, beta, fibre_weights, point_weights, point_hessian = (
                    func_particle_stencil_terms(
                        i_s, gradient_contact, hessian_contact, mpm_state, sparse_grid, mpm_info
                    )
                )
                for i_pair in range(STENCIL_PAIRS):
                    a = pairs_a[i_pair]
                    b = pairs_b[i_pair]
                    block_ab = func_stencil_pair_block(
                        a, b, r_frame, U, S_block, alpha, beta, fibre_weights, point_weights, point_hessian
                    )
                    if i_s == i_s_start:
                        func_write_triplet(
                            newton_rows.matrix, i_t_start + i_pair, slots[a], slots[b], block_ab, is_symmetric=True
                        )
                    else:
                        func_add_triplet(
                            newton_rows.matrix, i_t_start + i_pair, slots[a], slots[b], block_ab, is_symmetric=True
                        )
    else:
        qd.loop_config(block_dim=128)
        for i_s in range((n_valid + 31) // 32 * 32):
            tid = gs.qd_int(qd.simt.subgroup.invocation_id()) & 31
            is_valid = i_s < n_valid
            # A lane past the particles takes the terms of the last one and adds nothing, every lane taking part in the
            # sums of the warp
            i_s_clamped = qd.min(i_s, n_valid - 1)
            i_task = gs.qd_int(-1)
            if is_valid:
                i_task = mpm_state.sorted_particles.tasks_idx[i_s]
            i_task_prev, i_task_next = qd_slot_neighbors(tid, i_task)
            gradient_contact, hessian_contact = func_staged_contact_terms(i_s_clamped, mpm_state, mpm_config)
            slots, r_frame, U, S_block, alpha, beta, fibre_weights, point_weights, point_hessian = (
                func_particle_stencil_terms(
                    i_s_clamped, gradient_contact, hessian_contact, mpm_state, sparse_grid, mpm_info
                )
            )
            for i_pair in range(STENCIL_PAIRS):
                a = pairs_a[i_pair]
                b = pairs_b[i_pair]
                block_ab = gs.qd_float(is_valid) * func_stencil_pair_block(
                    a, b, r_frame, U, S_block, alpha, beta, fibre_weights, point_weights, point_hessian
                )
                is_tail = False
                for j, k in qd.static(qd.ndrange(3, 3)):
                    block_jk, is_tail = qd_segmented_sum(tid, i_task, i_task_prev, i_task_next, block_ab[j, k])
                    block_ab[j, k] = block_jk
                if is_tail:
                    func_write_triplet(
                        newton_rows.matrix,
                        n_nodes + STENCIL_PAIRS * i_task + i_pair,
                        slots[a],
                        slots[b],
                        block_ab,
                        is_symmetric=True,
                    )

    # The gradient of every node: its kinetic term and the factors staged by the particles of the cells around it
    N_LANES = qd.static(1 if gs.backend == gs.cpu else 32)
    n_particles = mpm_state.particles.pos.shape[0]
    inv_dx = mpm_info.inv_dx[None]
    dt = mpm_info.substep_dt[None]
    qd.loop_config(block_dim=128)
    for i_th in range(n_nodes * N_LANES):
        i_lane = gs.qd_int(0)
        if qd.static(N_LANES > 1):
            i_lane = gs.qd_int(qd.simt.subgroup.invocation_id()) & 31
        i_n = i_th // N_LANES
        i_b, block = func_block_of_key(sparse_grid.blocks_key[i_n // 64])
        node = block * 4 + func_local_cell(i_n % 64)
        gradient = qd.Vector.zero(gs.qd_float, 3)
        for a in range(8):
            offset = func_node_offset(a)
            cell = node - offset
            i_s_start, i_s_end = func_cell_particles(i_b, cell, i_lane, sparse_grid, is_lanes=N_LANES > 1)
            for i_chunk_ in range((i_s_end - i_s_start + N_LANES - 1) // N_LANES):
                i_s = i_s_start + i_chunk_ * N_LANES + i_lane
                if i_s < i_s_end:
                    i_p = mpm_state.sorted_particles.particles_idx[i_s] - i_b * n_particles
                    x_p = mpm_state.particles.pos[i_p, i_b]
                    w, grad_w = func_q1_node_weight(offset, x_p * inv_dx - cell.cast(gs.qd_float), inv_dx)
                    gradient += (
                        w * mpm_state.sorted_particles.point_gradient[i_s]
                        + mpm_state.sorted_particles.stress_factor[i_s] @ grad_w
                    )
        if qd.static(N_LANES > 1):
            for d in qd.static(range(3)):
                gradient[d] = qd_block_sum(gradient[d])
        if i_lane == 0:
            mass = sparse_grid.nodes_mass[i_n]
            gradient_node = qd.Vector.zero(gs.qd_float, 3)
            if sparse_grid.nodes_is_fixed[i_n] == 0:
                u = sparse_grid.nodes_u[i_n]
                gradient_node = gradient + mass * (u - dt * sparse_grid.nodes_vel_free[i_n])
            newton_rows.gradient[i_n] = gradient_node


@qd.func(requires_top_level=True)
def func_mpm_kinetic_energy(energies: qd.Tensor, envs_mask: qd.Tensor, sparse_grid: MPMSparseGrid, mpm_info: MPMInfo):
    """Add the kinetic energy of the nodes at their displacement into the entry of every environment selected, the nodes
    of an environment following one another (see 'func_node_kinetic_energy')."""
    n_nodes = sparse_grid.n_blocks[None] * 64
    qd.loop_config(block_dim=256)
    for i_n in range((n_nodes + 31) // 32 * 32):
        i_b_key = gs.qd_int(-1)
        energy = gs.qd_float(0.0)
        if i_n < n_nodes:
            i_b, block = func_block_of_key(sparse_grid.blocks_key[i_n // 64])
            if envs_mask[i_b]:
                i_b_key = i_b
                energy = func_node_kinetic_energy(
                    sparse_grid.nodes_mass[i_n],
                    sparse_grid.nodes_is_fixed[i_n],
                    sparse_grid.nodes_u[i_n],
                    sparse_grid.nodes_vel_free[i_n],
                    mpm_info.substep_dt[None],
                )
        func_add_by_key(i_b_key, energy, energies)


@qd.func(requires_top_level=True)
def func_mpm_precond_invert(newton_rows: NewtonRows):
    """Invert the diagonal blocks of the rows into their block Jacobi preconditioner, every block equilibrated to a unit
    diagonal, which keeps the inverse accurate for the rows of very different stiffness."""
    for i_entry in range(newton_rows.matrix.n_entries[None]):
        i_r, j_r, block = func_read_entry(newton_rows.matrix, i_entry, block_rows=3, block_cols=3)
        if i_r == j_r:
            diagonal = func_diagonal(block)
            scale = qd.select(diagonal > 0.0, 1.0 / qd.sqrt(qd.select(diagonal > 0.0, diagonal, 1.0)), 1.0)
            scaling = scale.outer_product(scale)
            inverse = qd.Matrix.identity(gs.qd_float, 3)
            if (block * scaling).determinant() > 0.0:
                inverse = (block * scaling).inverse()
            newton_rows.jacobi[i_r] = inverse * scaling


@qd.func(requires_top_level=True)
def func_mpm_matvec(newton_rows: NewtonRows, envs_mask: qd.Tensor, p_dot_ap: qd.Tensor, has_p_dot_ap: qd.template()):
    """
    Add the product Ap of the matrix of the rows with the search direction of every environment selected, the upper
    triangle standing for the transposed blocks of the lower one.

    With `has_p_dot_ap`, p . Ap of every environment adds to its entry in `p_dot_ap` along, in the energy form.
    """
    n_entries = newton_rows.matrix.n_entries[None]
    qd.loop_config(block_dim=256)
    for i_entry in range((n_entries + 31) // 32 * 32):
        i_r_key = gs.qd_int(-1)
        i_b_key = gs.qd_int(-1)
        product = qd.Vector.zero(gs.qd_float, 3)
        energy = gs.qd_float(0.0)
        if i_entry < n_entries:
            i_r, j_r, block = func_read_entry(newton_rows.matrix, i_entry, block_rows=3, block_cols=3)
            i_b = newton_rows.rows_env[i_r]
            if envs_mask[i_b]:
                i_r_key = i_r
                i_b_key = i_b
                p_col = newton_rows.p[j_r]
                product = block @ p_col
                energy = p_col.dot(product)
                if i_r != j_r:
                    p_row = newton_rows.p[i_r]
                    qd.atomic_add(newton_rows.Ap[j_r], block.transpose() @ p_row)
                    energy = 2.0 * p_row.dot(product)
        func_add_by_key(i_r_key, product, newton_rows.Ap, n_components=3)
        if qd.static(has_p_dot_ap):
            func_add_by_key(i_b_key, energy, p_dot_ap)


@qd.func(requires_top_level=True)
def func_mpm_init_line_search(
    newton_rows: NewtonRows,
    envs_mask: qd.Tensor,
    max_displacement: qd.Tensor,
    alpha: qd.Tensor,
    mpm_state: MPMState,
    sparse_grid: MPMSparseGrid,
    mpm_info: MPMInfo,
):
    """
    Set the Newton direction of the nodes of every environment selected, minus the solution of the linear solve, from
    their start point.

    The largest update of the displacement of a particle adds to the entry of its environment in `max_displacement`. The
    step of an environment in `alpha` stays short of the first zero volume of a liquid particle, which
    'func_particle_inversion_step_cap' finds along the direction.
    """
    for i_n in range(newton_rows.n_rows[None]):
        i_b = newton_rows.rows_env[i_n]
        if envs_mask[i_b]:
            sparse_grid.nodes_du[i_n] = -newton_rows.solution[i_n]
            sparse_grid.nodes_u_start[i_n] = sparse_grid.nodes_u[i_n]
    for i_s in range(mpm_state.sorted_particles.n_valid[None]):
        i_b, du = func_particle_displacement(i_s, sparse_grid.nodes_du, mpm_state, sparse_grid, mpm_info)
        if envs_mask[i_b]:
            qd.atomic_max(max_displacement[i_b], qd.max(qd.abs(du).max(), 0.0))
            is_capped, alpha_cap = func_particle_inversion_step_cap(i_s, mpm_state, sparse_grid, mpm_info)
            if is_capped:
                qd.atomic_min(alpha[i_b], alpha_cap)


@qd.func(requires_top_level=True)
def func_mpm_step_forward(sparse_grid: MPMSparseGrid, envs_mask: qd.Tensor, alpha: qd.Tensor):
    """
    Move the nodes of every environment selected to the trial point of its step along the Newton direction.

    The step of an environment is its entry in `alpha`.
    """
    n_rows = sparse_grid.n_blocks[None] * 64
    for i_n in range(n_rows):
        i_b, block = func_block_of_key(sparse_grid.blocks_key[i_n // 64])
        if envs_mask[i_b]:
            sparse_grid.nodes_u[i_n] = sparse_grid.nodes_u_start[i_n] + alpha[i_b] * sparse_grid.nodes_du[i_n]
