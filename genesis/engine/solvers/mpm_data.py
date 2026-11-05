"""Data of the implicit MPM solver.

The particle state keeps the layout of the solver's public fields. This module holds the data the implicit step adds:
the material table, the sparse grid rebuilt every substep, the Newton state over the grid displacement, and the linear
system the Newton solve assembles. The grid, Newton and linear-system structs are built at a capacity in blocks and
triplets. Growing a capacity builds a struct again and swaps it on the participant of the Newton solve that holds it
(see MpmNewtonParticipant).
"""

import numpy as np

import quadrants as qd

import genesis as gs
from genesis.engine.numerics.bcoo_matrix import BCOOMatrixScratch, bcoo_matrix_sizes, init_bcoo_matrix
from genesis.engine.numerics.block_reduce import REDUCE_BLOCK_DIM
from genesis.engine.numerics.dynamic_exclusive_sum import DynamicExclusiveSumScratch, dynamic_exclusive_sum_sizes
from genesis.engine.numerics.dynamic_radix_sort import DynamicRadixSortScratch, dynamic_radix_sort_sizes
from genesis.engine.numerics.linear_pcg import PCG_ROWS, PCGWorkspaceScratch
from genesis.engine.numerics.open_addressing_hash import OpenAddressingHashScratch, open_addressing_hash_sizes
from genesis.utils.data import ENVS, Data, DataKind, Size, field, requires

# Number of cells along each axis of a grid block, and of cells (and nodes) in a block
BLOCK_DIM = 4
BLOCK_CELLS = BLOCK_DIM**3

# Every assembled elastic task emits the upper triangle of the 8 x 8 node stencil of its cell
STENCIL_BLOCKS = 36

# Brick preconditioner: one-level additive Schwarz over bricks of 2 x 2 x 4 nodes of a grid block, each brick a dense
# 48 x 48 matrix stored as the packed upper triangle of its 16 x 16 node blocks
BRICK_NODES = 16
BRICKS_PER_BLOCK = BLOCK_CELLS // BRICK_NODES
BRICK_SYM_BLOCKS = BRICK_NODES * (BRICK_NODES + 1) // 2
BRICK_SCALARS = BRICK_SYM_BLOCKS * 9
BRICK_DIM = BRICK_NODES * 3
# Bricks inverted per thread block of the inversion, one warp each
BRICK_INVERT_WARPS = 4

# Soft contact rows of a particle: one domain-wall row per axis, then the pyramid rows of its deepest Rigid contact
WALL_ROWS = 3
PYRAMID_ROWS = 4
CONTACT_ROWS = WALL_ROWS + PYRAMID_ROWS

# Sizes of the data of the implicit step that grows: the sorted particle keys, the blocks of the grid and its particle
# blocks, the nodes of the blocks, the keys of the neighbor blocks of the particle blocks, and the bricks of the nodes
SORT = Size("sort")
BLOCKS = Size("blocks")
PARTICLE_BLOCKS = Size("particle_blocks")
NODES = BLOCKS * BLOCK_CELLS
NEIGHBORS = 8 * PARTICLE_BLOCKS
BRICKS = NODES // BRICK_NODES
# Entries of the scratch of the serial brick inversion of a CPU backend
BRICK_SCRATCH = Size("brick_scratch")

# Sizes of the particle data: the frames of the substep window and the particles
FRAMES = Size("frames")
PARTICLES = Size("particles")
# Sizes of the material table: the materials and the parameters of each
MATERIALS = Size("materials")
MATERIAL_PARAMS = Size("material_params")


class MpmInfo(Data, kind=DataKind.CONSTANT, shape=()):
    """Scene values of the implicit material point method (MPM) kernels, as device scalars.

    A scene of another size or resolution runs the kernels compiled for the previous one.
    """

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
    """Parameter-major material table, `material_table[param * n_materials + slot]`, and the energy family and the
    return map of each slot."""

    material_table: qd.Tensor = field(float, shape=(MATERIAL_PARAMS * MATERIALS,))
    mat_family: qd.Tensor = field(int, shape=(MATERIALS,))
    mat_plastic: qd.Tensor = field(int, shape=(MATERIALS,))


class MpmErrno(Data, kind=DataKind.STATE, shape=(ENVS,)):
    """Error flags of the implicit material point method (MPM) solver per environment.

    The kernels building and reading the grid raise them instead of indexing out of bounds (see
    mpm_func.flag_env_error).
    """

    errno: qd.Tensor = field(int)


def _padded(length: int, multiple: int = REDUCE_BLOCK_DIM) -> int:
    return max(((max(length, 1) + multiple - 1) // multiple) * multiple, multiple)


class MpmGridScratch(Data, kind=DataKind.SCRATCH):
    """Block-sparse grid rebuilt from the particles every substep.

    Particles are sorted by (environment, block, cell). Every particle block activates its positive-octant neighbour
    blocks as node storage. A block slot is the rank of its key among the sorted unique keys, and a node or cell slot is
    `block_slot * BLOCK_CELLS + local index`. The capacities in blocks and in sorted particles are the sizes of the
    arrays (see build_mpm_grid).
    """

    errors: MpmErrno = requires()
    # Number of particle keys sorted, n_envs * n_particles
    n_sort_keys: qd.Tensor = field(int, ndarray=True)
    n_valid: qd.Tensor = field(int, ndarray=True)
    n_valid_chunks: qd.Tensor = field(int, ndarray=True)
    n_valid_reduce_blocks: qd.Tensor = field(int, ndarray=True)
    n_particle_blocks: qd.Tensor = field(int, ndarray=True)
    n_neighbor_keys: qd.Tensor = field(int, ndarray=True)
    n_blocks: qd.Tensor = field(int, ndarray=True)
    n_node_reduce_blocks: qd.Tensor = field(int, ndarray=True)
    n_active_cells: qd.Tensor = field(int, ndarray=True)
    n_tasks: qd.Tensor = field(int, ndarray=True)
    has_grid_overflow: qd.Tensor = field(int, ndarray=True)
    grid_demand: qd.Tensor = field(int, ndarray=True)

    sort_keys: qd.Tensor = field(qd.u64, shape=(SORT,), ndarray=True)
    sort_keys_tmp: qd.Tensor = field(qd.u64, shape=(SORT,), ndarray=True)
    sort_ids: qd.Tensor = field(int, shape=(SORT,), ndarray=True)
    sort_ids_tmp: qd.Tensor = field(int, shape=(SORT,), ndarray=True)
    particle_sort: DynamicRadixSortScratch

    pblock_keys: qd.Tensor = field(qd.u64, shape=(PARTICLE_BLOCKS,), ndarray=True)
    neighbor_keys: qd.Tensor = field(qd.u64, shape=(NEIGHBORS,), ndarray=True)
    neighbor_keys_tmp: qd.Tensor = field(qd.u64, shape=(NEIGHBORS,), ndarray=True)
    neighbor_perm: qd.Tensor = field(int, shape=(NEIGHBORS,), ndarray=True)
    neighbor_perm_tmp: qd.Tensor = field(int, shape=(NEIGHBORS,), ndarray=True)
    block_sort: DynamicRadixSortScratch
    block_flags: qd.Tensor = field(int, shape=(NEIGHBORS,), ndarray=True)
    block_ranks: qd.Tensor = field(int, shape=(NEIGHBORS,), ndarray=True)
    block_scan: DynamicExclusiveSumScratch
    block_coords: qd.Tensor = field(qd.u64, shape=(BLOCKS,), ndarray=True)
    hash: OpenAddressingHashScratch

    cell_flags: qd.Tensor = field(int, shape=(SORT,), ndarray=True)
    cell_ranks: qd.Tensor = field(int, shape=(SORT,), ndarray=True)
    cell_scan: DynamicExclusiveSumScratch
    cell_starts: qd.Tensor = field(int, shape=(SORT + 1,), ndarray=True)
    task_flags: qd.Tensor = field(int, shape=(SORT,), ndarray=True)
    task_ranks: qd.Tensor = field(int, shape=(SORT,), ndarray=True)
    task_scan: DynamicExclusiveSumScratch
    cell_seg_start: qd.Tensor = field(int, shape=(NODES,), ndarray=True)
    cell_seg_end: qd.Tensor = field(int, shape=(NODES,), ndarray=True)

    node_mass: qd.Tensor = field(float, shape=(NODES,), ndarray=True)
    # Node touched by a non-free particle, whose displacement the Newton solve holds at zero
    node_fixed: qd.Tensor = field(int, shape=(NODES,), ndarray=True)
    # Diagonal 3x3 block of every node of the physical Hessian (kinetic and elastic), from which the inverse weights of
    # the particles in their contact rows are computed once (see func_contact_invweight)
    node_diag: qd.Tensor = field((float, (3, 3)), shape=(NODES,), ndarray=True)
    node_v: qd.Tensor = field((float, 3), shape=(NODES,), ndarray=True)
    node_v_free: qd.Tensor = field((float, 3), shape=(NODES,), ndarray=True)


class MpmNewtonScratch(Data, kind=DataKind.SCRATCH):
    """Newton state of the grid displacement `u`, the per-particle staging of the assembly, and the soft contact rows
    of the particles."""

    u: qd.Tensor = field(float, shape=(3 * NODES,), ndarray=True)
    u_temp: qd.Tensor = field(float, shape=(3 * NODES,), ndarray=True)
    du: qd.Tensor = field(float, shape=(3 * NODES,), ndarray=True)
    # Adjoint of the node momentum and mass of the particle-to-grid transfer, which the backward pass of a substep
    # gathers to the particles
    adj_mom: qd.Tensor = field((float, 3), shape=(NODES,), ndarray=True)
    adj_mass: qd.Tensor = field(float, shape=(NODES,), ndarray=True)
    # Per sorted particle: V dt^2 P(F_trial) F^T, the factor of its elastic gradient contribution to every node
    g_p: qd.Tensor = field((float, (3, 3)), shape=(SORT,), ndarray=True)
    # Per sorted particle: the gradient of its contact rows with respect to its trial displacement sum_a w_a u_a
    g_contact: qd.Tensor = field((float, 3), shape=(SORT,), ndarray=True)
    # Per sorted particle: the gradient and Hessian of its Rigid contact rows alone, which couple it with the dofs of
    # its link, and the displacement of its contact point on the link over the substep
    g_rigid: qd.Tensor = field((float, 3), shape=(SORT,), ndarray=True)
    h_rigid: qd.Tensor = field((float, (3, 3)), shape=(SORT,), ndarray=True)
    contact_dr: qd.Tensor = field((float, 3), shape=(SORT,), ndarray=True)
    # Per sorted particle: the spring that attaches it to its target (see func_build_constraint_rows), its stiffness
    # dt^2 k and damping dt c, both zero for a free particle, and the target. They start at zero, since the substeps
    # build them only once a constraint exists.
    constraint_spring: qd.Tensor = field(float, shape=(SORT,), ndarray=True)
    constraint_damping: qd.Tensor = field(float, shape=(SORT,), ndarray=True)
    constraint_target: qd.Tensor = field((float, 3), shape=(SORT,), ndarray=True)
    # Soft contact rows, CONTACT_ROWS per sorted particle at CONTACT_ROWS s + r, on its trial displacement dx =
    # sum_a w_a u_a: direction J, stiffness D (zero where there is no row) and target t of the row term
    # 1/2 D min(0, J.dx - t)^2, and the Rigid link of the particle's pyramid rows (-1 if none)
    contact_dir: qd.Tensor = field((float, 3), shape=(CONTACT_ROWS * SORT,), ndarray=True)
    contact_D: qd.Tensor = field(float, shape=(CONTACT_ROWS * SORT,), ndarray=True)
    contact_t: qd.Tensor = field(float, shape=(CONTACT_ROWS * SORT,), ndarray=True)
    contact_link: qd.Tensor = field(int, shape=(SORT,), ndarray=True)
    # The FEM surface triangle of the particle's pyramid rows (-1 if none), and the barycentric coordinates of its
    # closest point on it
    contact_tri: qd.Tensor = field(int, shape=(SORT,), ndarray=True)
    contact_bary: qd.Tensor = field((float, 3), shape=(SORT,), ndarray=True)
    # The PBD particle of the particle's pyramid rows, in PBD's reordered layout (-1 if none)
    contact_pbd: qd.Tensor = field(int, shape=(SORT,), ndarray=True)
    # Nearest obstacle of each sorted particle during a bounding volume hierarchy query: its gap, its index, and, for a
    # triangle, the distance to it and the alignment of its normal with the direction to the particle (which together
    # pick the triangle), the barycentric coordinates of the closest point and the contact normal. The query point and
    # the frame at which the obstacles are read sit alongside, for the filter measuring the gaps.
    query_phi: qd.Tensor = field(float, shape=(SORT,), ndarray=True)
    query_idx: qd.Tensor = field(int, shape=(SORT,), ndarray=True)
    query_dist: qd.Tensor = field(float, shape=(SORT,), ndarray=True)
    query_align: qd.Tensor = field(float, shape=(SORT,), ndarray=True)
    query_bary: qd.Tensor = field((float, 3), shape=(SORT,), ndarray=True)
    query_normal: qd.Tensor = field((float, 3), shape=(SORT,), ndarray=True)
    query_pos: qd.Tensor = field((float, 3), shape=(SORT,), ndarray=True)
    query_frame: qd.Tensor = field(int, ndarray=True)


class MpmLinearScratch(Data, kind=DataKind.SCRATCH):
    """Block-sparse Newton matrix over 3x3 node blocks, its right-hand side, solution, preconditioner and conjugate
    gradient state."""

    matrix: BCOOMatrixScratch
    pcg: PCGWorkspaceScratch
    b_rhs: qd.Tensor = field(float, shape=(3 * NODES,), ndarray=True)
    solution: qd.Tensor = field(float, shape=(3 * NODES,), ndarray=True)
    # Brick preconditioner: per brick, its packed 48 x 48 matrix and then inverse, and the mask of its nodes coupled to
    # another node of the brick
    brick: qd.Tensor = field(float, shape=(BRICKS * BRICK_SCALARS,), ndarray=True)
    brick_mask: qd.Tensor = field(int, shape=(BRICKS,), ndarray=True)
    # The serial inversion of a CPU backend: per brick, the equilibration of its rows and the pivot column
    brick_scratch: qd.Tensor = field(float, shape=(BRICK_SCRATCH,), ndarray=True)


def build_mpm_grid(n_sort: int, max_blocks: int) -> MpmGridScratch:
    """Build the grid of a capacity of sorted particles and blocks, with every counter at zero.

    `n_sort` is the number of sorted particles and `max_blocks` the capacity in blocks. The error flags of the solver
    must be built in the entered namespace (see `requires`).
    """
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
        grid.n_valid_reduce_blocks,
        grid.n_particle_blocks,
        grid.n_neighbor_keys,
        grid.n_blocks,
        grid.n_node_reduce_blocks,
        grid.n_active_cells,
        grid.n_tasks,
        grid.has_grid_overflow,
        grid.grid_demand,
    ):
        counter.fill(0)
    return grid


def build_mpm_newton(n_sort: int, max_blocks: int) -> MpmNewtonScratch:
    """Build the Newton state of a capacity of sorted particles and blocks, with the particle constraints at zero.

    `n_sort` is the number of sorted particles and `max_blocks` the capacity in blocks.
    """
    newton = MpmNewtonScratch(sizes={SORT: _padded(n_sort), BLOCKS: max_blocks})
    for constraint in (newton.constraint_spring, newton.constraint_damping, newton.constraint_target):
        constraint.fill(0.0)
    return newton


def build_mpm_linear(max_blocks: int, max_triplets: int) -> MpmLinearScratch:
    """Build the linear system of a capacity of blocks and triplets, with no live row and no live triplet.

    `max_blocks` is the capacity in blocks and `max_triplets` the capacity in triplets.
    """
    max_nodes = max_blocks * BLOCK_CELLS
    n_bricks = max(max_nodes // BRICK_NODES, 1)
    linear = MpmLinearScratch(
        sizes={
            BLOCKS: max_blocks,
            BRICK_SCRATCH: n_bricks * 2 * BRICK_DIM if gs.backend == gs.cpu else 1,
            MpmLinearScratch.matrix: bcoo_matrix_sizes(max_triplets, block_shape=(3, 3)),
            MpmLinearScratch.pcg: {PCG_ROWS: max_nodes},
        }
    )
    init_bcoo_matrix(linear.matrix, n_triplets=0)
    linear.pcg.n_rows.fill(0)
    linear.pcg.row_env.fill(0)
    return linear


@qd.data_oriented
class MpmForceFields:
    """The force fields acting on the MPM grid, read with a static loop over the list."""

    def __init__(self, force_fields) -> None:
        self.force_fields = list(force_fields)


# Slack a capacity gets when it grows on overflow: the new capacity is the demand times this factor
CAPACITY_GROW_FACTOR = 1.2
# Smallest block capacity, and the capacity before the first substep sizes the data to the particles
MIN_BLOCKS = 64


def triplet_capacity(max_blocks: int, n_tasks: int) -> int:
    """Count the triplets of one kinetic diagonal per node and STENCIL_BLOCKS per elastic task."""
    return max_blocks * BLOCK_CELLS + STENCIL_BLOCKS * n_tasks


def initial_capacities(positions: np.ndarray, inv_dx: float) -> tuple[int, int]:
    """Compute the block and triplet capacities from the particle positions, at twice the initial demand.

    `positions` has shape (n_envs, n_particles, 3). The blocks are those of the particles and their positive-octant
    neighbours. The elastic tasks are bounded by the active cells plus one per 32 particles.
    """
    n_envs, n_particles = positions.shape[:2]
    n_sort = n_envs * n_particles
    cells = np.floor(positions * inv_dx).astype(np.int64)
    env = np.broadcast_to(np.arange(n_envs, dtype=np.int64)[:, None, None], (n_envs, n_particles, 1))
    env_cells = np.concatenate((env, cells), axis=-1).reshape(-1, 4)
    n_active_cells = len(np.unique(env_cells, axis=0))
    particle_blocks = np.unique(
        np.concatenate((env_cells[:, :1], np.floor_divide(env_cells[:, 1:], BLOCK_DIM)), axis=1), axis=0
    )
    offsets = np.array([[0, (o >> 2) & 1, (o >> 1) & 1, o & 1] for o in range(8)], dtype=np.int64)
    n_active_blocks = len(np.unique((particle_blocks[:, None, :] + offsets[None]).reshape(-1, 4), axis=0))
    max_blocks = max(2 * n_active_blocks, MIN_BLOCKS)
    n_tasks = min(max(2 * (n_active_cells + n_sort // 32), MIN_BLOCKS), n_sort + n_sort // 32 + 1)
    return max_blocks, triplet_capacity(max_blocks, n_tasks)
