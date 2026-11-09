"""Data of the implicit material point method (MPM) solver: the material table, the sparse grid rebuilt every substep,
and the Newton state over the grid displacement."""

import numpy as np

import quadrants as qd

from genesis.engine.numerics.dynamic_exclusive_sum import DynamicExclusiveSumScratch, dynamic_exclusive_sum_sizes
from genesis.engine.numerics.dynamic_radix_sort import DynamicRadixSortScratch, dynamic_radix_sort_sizes
from genesis.engine.numerics.open_addressing_hash import OpenAddressingHashScratch, open_addressing_hash_sizes
from genesis.utils.data import ENVS, Data, DataKind, Size, field, requires

from .newton_linear import PAIRS

# Number of cells along each axis of a grid block, and of cells (and nodes) in a block
BLOCK_DIM = 4
BLOCK_CELLS = BLOCK_DIM**3

# Threads of a block of the reductions by environment, to which the sorted particles are padded
REDUCE_BLOCK = 256

# Every assembled elastic task emits the upper triangle of the 8 x 8 node stencil of its cell
STENCIL_BLOCKS = 36

# Rows of the friction pyramid of a contact pair: n +- mu t1 and n +- mu t2, t1 and t2 two tangents of the normal n
# (see pyramid_directions)
PYRAMID_ROWS = 4

# Sizes of the data of the implicit step that grows: the sorted particle keys, the blocks of the grid and its particle
# blocks, the nodes of the blocks and the keys of the neighbor blocks of the particle blocks
SORT = Size("sort")
BLOCKS = Size("blocks")
PARTICLE_BLOCKS = Size("particle_blocks")
NODES = BLOCKS * BLOCK_CELLS
NEIGHBORS = 8 * PARTICLE_BLOCKS

# Sizes of the particle data: the frames of the substep window and the particles
FRAMES = Size("frames")
PARTICLES = Size("particles")
# Sizes of the material table: the materials and the parameters of each
MATERIALS = Size("materials")
MATERIAL_PARAMS = Size("material_params")


class MpmInfo(Data, kind=DataKind.CONSTANT, shape=()):
    """Scene values of the implicit material point method (MPM) kernels, as device scalars, so that a scene of another
    size or resolution runs the kernels compiled for the previous one."""

    n_envs: qd.Tensor = field(int, ndarray=True)
    n_particles: qd.Tensor = field(int, ndarray=True)
    substep_dt: qd.Tensor = field(float, ndarray=True)
    dx: qd.Tensor = field(float, ndarray=True)
    inv_dx: qd.Tensor = field(float, ndarray=True)
    particle_volume: qd.Tensor = field(float, ndarray=True)
    mass_scale: qd.Tensor = field(float, ndarray=True)
    n_materials: qd.Tensor = field(int, ndarray=True)
    # Width in volume ratio of the band below J = 1 over which the stiffness of the liquid rises (see
    # liquid_volume_terms)
    liquid_band: qd.Tensor = field(float, ndarray=True)
    # The solref / solimp parameters of the wall rows, in Rigid's form: timeconst, dampratio, dmin, dmax, width, mid,
    # power
    wall_sol_params: qd.Tensor = field((float, 7), ndarray=True)
    # Planes of the domain walls, `CubeBoundary.lower` / `upper`
    wall_lower: qd.Tensor = field((float, 3), ndarray=True)
    wall_upper: qd.Tensor = field((float, 3), ndarray=True)


class MpmParticles(Data, kind=DataKind.STATE, shape=(FRAMES, PARTICLES, ENVS)):
    """Particle state of every frame of the substep window."""

    pos: qd.Tensor = field((float, 3), grad=True)
    vel: qd.Tensor = field((float, 3), grad=True)
    C: qd.Tensor = field((float, (3, 3)), grad=True)
    F: qd.Tensor = field((float, (3, 3)), grad=True)
    actu: qd.Tensor = field(float, grad=True)
    Jp: qd.Tensor = field(float, grad=True)


class MpmParticlesNG(Data, kind=DataKind.STATE, shape=(FRAMES, PARTICLES, ENVS)):
    """Particle state without gradient."""

    active: qd.Tensor = field(bool)


class MpmParticlesInfo(Data, kind=DataKind.INFO, shape=(PARTICLES,)):
    """Static per-particle data."""

    material_idx: qd.Tensor = field(int)
    mass: qd.Tensor = field(float)
    default_Jp: qd.Tensor = field(float)
    free: qd.Tensor = field(bool)
    muscle_group: qd.Tensor = field(int)
    muscle_direction: qd.Tensor = field((float, 3))
    # Inverse weight matrix of the particle in its contact rows, a model constant (see func_contact_invweight)
    contact_invweight: qd.Tensor = field((float, (3, 3)))


class MpmMaterialTable(Data, kind=DataKind.CONSTANT):
    """Parameter-major material table, `material_table[param * n_materials + i_mat]`, and the energy family and the
    return map of each material."""

    material_table: qd.Tensor = field(float, shape=(MATERIAL_PARAMS * MATERIALS,))
    materials_family: qd.Tensor = field(int, shape=(MATERIALS,))
    materials_plastic: qd.Tensor = field(int, shape=(MATERIALS,))


class MpmErrno(Data, kind=DataKind.STATE, shape=(ENVS,)):
    """Error flags of the implicit material point method (MPM) solver per environment, which the kernels building and
    reading the grid raise instead of indexing out of bounds (see mpm_func.flag_env_error)."""

    errno: qd.Tensor = field(int)


def _padded(length: int, multiple: int = REDUCE_BLOCK) -> int:
    return max(((max(length, 1) + multiple - 1) // multiple) * multiple, multiple)


class MpmGridScratch(Data, kind=DataKind.SCRATCH):
    """Block-sparse grid rebuilt from the particles every substep."""

    errors: MpmErrno = requires()
    # Number of particle keys sorted, n_envs * n_particles
    n_sort_keys: qd.Tensor = field(int, ndarray=True)
    n_valid: qd.Tensor = field(int, ndarray=True)
    n_valid_chunks: qd.Tensor = field(int, ndarray=True)
    n_particle_blocks: qd.Tensor = field(int, ndarray=True)
    n_neighbor_keys: qd.Tensor = field(int, ndarray=True)
    n_blocks: qd.Tensor = field(int, ndarray=True)
    n_active_cells: qd.Tensor = field(int, ndarray=True)
    n_tasks: qd.Tensor = field(int, ndarray=True)
    has_grid_overflow: qd.Tensor = field(int, ndarray=True)
    grid_demand: qd.Tensor = field(int, ndarray=True)

    sort_keys: qd.Tensor = field(qd.u64, shape=(SORT,), ndarray=True)
    sort_keys_buffer: qd.Tensor = field(qd.u64, shape=(SORT,), ndarray=True)
    sort_ids: qd.Tensor = field(int, shape=(SORT,), ndarray=True)
    sort_ids_buffer: qd.Tensor = field(int, shape=(SORT,), ndarray=True)
    particle_sort: DynamicRadixSortScratch

    particle_blocks_key: qd.Tensor = field(qd.u64, shape=(PARTICLE_BLOCKS,), ndarray=True)
    neighbors_key: qd.Tensor = field(qd.u64, shape=(NEIGHBORS,), ndarray=True)
    neighbors_key_buffer: qd.Tensor = field(qd.u64, shape=(NEIGHBORS,), ndarray=True)
    neighbors_source_idx: qd.Tensor = field(int, shape=(NEIGHBORS,), ndarray=True)
    neighbors_source_idx_buffer: qd.Tensor = field(int, shape=(NEIGHBORS,), ndarray=True)
    block_sort: DynamicRadixSortScratch
    neighbors_is_new_block: qd.Tensor = field(int, shape=(NEIGHBORS,), ndarray=True)
    neighbors_block_idx: qd.Tensor = field(int, shape=(NEIGHBORS,), ndarray=True)
    block_scan: DynamicExclusiveSumScratch
    blocks_key: qd.Tensor = field(qd.u64, shape=(BLOCKS,), ndarray=True)
    hash: OpenAddressingHashScratch

    sort_is_cell_start: qd.Tensor = field(int, shape=(SORT,), ndarray=True)
    sort_cell_idx: qd.Tensor = field(int, shape=(SORT,), ndarray=True)
    cell_scan: DynamicExclusiveSumScratch
    cells_start: qd.Tensor = field(int, shape=(SORT + 1,), ndarray=True)
    sort_is_task_start: qd.Tensor = field(int, shape=(SORT,), ndarray=True)
    sort_task_idx: qd.Tensor = field(int, shape=(SORT,), ndarray=True)
    task_scan: DynamicExclusiveSumScratch
    cells_segment_start: qd.Tensor = field(int, shape=(NODES,), ndarray=True)
    cells_segment_end: qd.Tensor = field(int, shape=(NODES,), ndarray=True)

    nodes_mass: qd.Tensor = field(float, shape=(NODES,), ndarray=True)
    # Node touched by a non-free particle, whose displacement the Newton solve holds at zero
    nodes_is_fixed: qd.Tensor = field(int, shape=(NODES,), ndarray=True)
    # Diagonal 3x3 block of every node of the physical Hessian (kinetic and elastic), from which the inverse weights of
    # the particles in their contact rows are computed once (see func_contact_invweight)
    nodes_diagonal: qd.Tensor = field((float, (3, 3)), shape=(NODES,), ndarray=True)
    nodes_vel: qd.Tensor = field((float, 3), shape=(NODES,), ndarray=True)
    nodes_vel_free: qd.Tensor = field((float, 3), shape=(NODES,), ndarray=True)


class MpmNewtonScratch(Data, kind=DataKind.SCRATCH):
    """Newton state of the grid displacement `u` and the per-particle staging of the assembly."""

    u: qd.Tensor = field((float, 3), shape=(NODES,), ndarray=True)
    u_start: qd.Tensor = field((float, 3), shape=(NODES,), ndarray=True)
    du: qd.Tensor = field((float, 3), shape=(NODES,), ndarray=True)
    # Adjoint of the node momentum and mass of the particle-to-grid transfer, which the backward pass of a substep
    # gathers to the particles
    nodes_momentum_bar: qd.Tensor = field((float, 3), shape=(NODES,), ndarray=True)
    nodes_mass_bar: qd.Tensor = field(float, shape=(NODES,), ndarray=True)
    # Per sorted particle: V dt^2 P(F_trial) F^T, the factor of its elastic gradient contribution to every node
    particles_stress_factor: qd.Tensor = field((float, (3, 3)), shape=(SORT,), ndarray=True)
    # Per sorted particle: its position at the substep start, which the contact stages read for its stencil inside
    # the graph of the conjugate gradient
    # FIXME: quadrants#946 - a graph refuses the ndarrays that own a gradient, as the particle positions do
    particles_pos: qd.Tensor = field((float, 3), shape=(SORT,), ndarray=True)
    # Per sorted particle: the gradient of its constraint spring with respect to its trial displacement sum_a w_a u_a
    particles_spring_grad: qd.Tensor = field((float, 3), shape=(SORT,), ndarray=True)
    # Per sorted particle: the spring that attaches it to its target (see func_build_constraint_rows), its stiffness
    # dt^2 k and damping dt c, both zero for a free particle, and the target. They start at zero, since the substeps
    # build them only once a constraint exists.
    constraint_spring: qd.Tensor = field(float, shape=(SORT,), ndarray=True)
    constraint_damping: qd.Tensor = field(float, shape=(SORT,), ndarray=True)
    constraint_target: qd.Tensor = field((float, 3), shape=(SORT,), ndarray=True)


class MpmWallPairs(Data, kind=DataKind.SCRATCH):
    """The contact pairs of the particles with the domain walls (see func_build_wall_pairs): a soft row on the trial
    displacement of sorted particle `sorted_idx` along the inward normal `sign` e_axis, of stiffness `stiffness` and
    target `target`. `n_pairs` counts the pairs asked for, above the capacity on overflow."""

    n_pairs: qd.Tensor = field(int, ndarray=True)
    sorted_idx: qd.Tensor = field(int, shape=(PAIRS,), ndarray=True)
    axis: qd.Tensor = field(int, shape=(PAIRS,), ndarray=True)
    sign: qd.Tensor = field(float, shape=(PAIRS,), ndarray=True)
    stiffness: qd.Tensor = field(float, shape=(PAIRS,), ndarray=True)
    target: qd.Tensor = field(float, shape=(PAIRS,), ndarray=True)


def build_mpm_grid(n_sort: int, max_blocks: int) -> MpmGridScratch:
    """Build the grid of `n_sort` sorted particles and `max_blocks` blocks, with every counter at zero, the error flags
    of the solver being bound from the entered namespace (see `requires`)."""
    n_sort_padded = _padded(n_sort)
    n_neighbors = 8 * max_blocks
    grid = MpmGridScratch(
        sizes={
            SORT: n_sort_padded,
            BLOCKS: max_blocks,
            PARTICLE_BLOCKS: max_blocks,
            MpmGridScratch.particle_sort: dynamic_radix_sort_sizes(key_bits=64, capacity=n_sort_padded),
            MpmGridScratch.block_sort: dynamic_radix_sort_sizes(key_bits=64, capacity=n_neighbors),
            MpmGridScratch.block_scan: dynamic_exclusive_sum_sizes(n_neighbors),
            MpmGridScratch.hash: open_addressing_hash_sizes(max_blocks),
            MpmGridScratch.cell_scan: dynamic_exclusive_sum_sizes(n_sort_padded),
            MpmGridScratch.task_scan: dynamic_exclusive_sum_sizes(n_sort_padded),
        }
    )
    for counter in (
        grid.n_sort_keys,
        grid.n_valid,
        grid.n_valid_chunks,
        grid.n_particle_blocks,
        grid.n_neighbor_keys,
        grid.n_blocks,
        grid.n_active_cells,
        grid.n_tasks,
        grid.has_grid_overflow,
        grid.grid_demand,
    ):
        counter.fill(0)
    return grid


def build_mpm_newton(n_sort: int, max_blocks: int) -> MpmNewtonScratch:
    """Build the Newton state of `n_sort` sorted particles and `max_blocks` blocks, with the particle constraints at
    zero."""
    newton = MpmNewtonScratch(sizes={SORT: _padded(n_sort), BLOCKS: max_blocks})
    for constraint in (newton.constraint_spring, newton.constraint_damping, newton.constraint_target):
        constraint.fill(0.0)
    return newton


def build_wall_pairs(max_pairs: int) -> MpmWallPairs:
    """Build the wall pairs of a capacity, with no pair."""
    pairs = MpmWallPairs(sizes={PAIRS: max_pairs})
    pairs.n_pairs.fill(0)
    return pairs


@qd.data_oriented
class MpmForceFields:
    """The force fields acting on the material point method (MPM) grid, read with a static loop over the list."""

    def __init__(self, force_fields) -> None:
        self.force_fields = list(force_fields)


# Smallest block capacity, and the capacity before the first substep sizes the data to the particles
MIN_BLOCKS = 64


def initial_block_capacity(positions: np.ndarray, inv_dx: float) -> int:
    """Compute the block capacity, twice the blocks of the particle positions [n_envs, n_particles, 3] and of their
    positive-octant neighbours."""
    n_envs, n_particles = positions.shape[:2]
    cells = np.floor(positions * inv_dx).astype(np.int64)
    env = np.broadcast_to(np.arange(n_envs, dtype=np.int64)[:, None, None], (n_envs, n_particles, 1))
    env_cells = np.concatenate((env, cells), axis=-1).reshape(-1, 4)
    particle_blocks = np.unique(
        np.concatenate((env_cells[:, :1], np.floor_divide(env_cells[:, 1:], BLOCK_DIM)), axis=1), axis=0
    )
    offsets = np.array([[0, (o >> 2) & 1, (o >> 1) & 1, o & 1] for o in range(8)], dtype=np.int64)
    n_active_blocks = len(np.unique((particle_blocks[:, None, :] + offsets[None]).reshape(-1, 4), axis=0))
    return max(2 * n_active_blocks, MIN_BLOCKS)
