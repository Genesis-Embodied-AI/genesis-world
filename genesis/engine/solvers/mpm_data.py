"""Data of the implicit MPM solver.

The particle state keeps the layout of the solver's public fields. What the implicit step adds lives here: the material
table, the sparse grid rebuilt every substep, the Newton state over the grid displacement, and the linear system the
Newton solve assembles. The grid, Newton and linear-system roots are mutable data-oriented objects, so that growing a
capacity replaces an array on an existing path while every kernel keeps the same object.
"""

import dataclasses
from typing import ClassVar

import numpy as np
import quadrants as qd

import genesis as gs
import genesis.utils.array_class as array_class
from genesis.engine.numerics.bcoo_matrix import BCOOMatrix
from genesis.engine.numerics.block_reduce import REDUCE_BLOCK_DIM
from genesis.engine.numerics.dynamic_exclusive_sum import DynamicExclusiveSum
from genesis.engine.numerics.dynamic_radix_sort import DynamicRadixSort
from genesis.engine.numerics.linear_pcg import PCGWorkspace
from genesis.engine.numerics.newton_control import NewtonControl
from genesis.engine.numerics.open_addressing_hash import OpenAddressingHash

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
# Bricks inverted per block of the inversion, one warp each
BRICK_INVERT_WARPS = 4


class MpmStaticConfig(metaclass=array_class.AutoInitMeta):
    """Compile-time configuration of the implicit MPM kernels."""

    n_envs: int
    n_particles: int
    substep_dt: float
    dx: float
    inv_dx: float
    particle_volume: float
    mass_scale: float
    n_materials: int


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class MpmMaterialTable:
    """Parameter-major material table, `material_table[param * n_materials + slot]`, and the family of each slot."""

    kind: ClassVar[array_class.DataKind] = array_class.DataKind.CONSTANT

    material_table: qd.Tensor
    mat_family: qd.Tensor


def get_mpm_material_table(table: np.ndarray, family: np.ndarray) -> MpmMaterialTable:
    material_table = array_class.V(dtype=gs.qd_float, shape=(max(table.size, 1),))
    mat_family = array_class.V(dtype=gs.qd_int, shape=(max(family.size, 1),))
    material_table.from_numpy(np.resize(table.astype(gs.np_float), max(table.size, 1)))
    mat_family.from_numpy(np.resize(family.astype(gs.np_int), max(family.size, 1)))
    return MpmMaterialTable(material_table=material_table, mat_family=mat_family)


def _scalar(dtype, value=0):
    result = qd.ndarray(dtype, shape=())
    np_dtype = {qd.i32: np.int32, qd.f32: np.float32, qd.f64: np.float64}[dtype]
    result.from_numpy(np.array(value, dtype=np_dtype))
    return result


def _padded(length: int, multiple: int = REDUCE_BLOCK_DIM) -> int:
    return max(((max(length, 1) + multiple - 1) // multiple) * multiple, multiple)


@qd.data_oriented
class MpmGrid:
    """Block-sparse grid rebuilt from the particles every substep.

    Particles are sorted by (environment, block, cell); every particle block activates its positive-octant neighbour
    blocks as node storage. A block slot is the rank of its key among the sorted unique keys, and a node or cell slot is
    `block_slot * BLOCK_CELLS + local index`.
    """

    def __init__(self, n_sort: int, max_blocks: int, max_particle_blocks: int) -> None:
        self.allocate(n_sort, max_blocks, max_particle_blocks)
        self.n_valid = _scalar(qd.i32)
        self.n_valid_chunks = _scalar(qd.i32)
        self.n_valid_reduce_blocks = _scalar(qd.i32)
        self.n_particle_blocks = _scalar(qd.i32)
        self.n_neighbor_keys = _scalar(qd.i32)
        self.n_blocks = _scalar(qd.i32)
        self.n_node_reduce_blocks = _scalar(qd.i32)
        self.n_active_cells = _scalar(qd.i32)
        self.n_tasks = _scalar(qd.i32)
        self.grid_overflow = _scalar(qd.i32)
        self.grid_demand = _scalar(qd.i32)

    def allocate(self, n_sort: int, max_blocks: int, max_particle_blocks: int) -> None:
        n_sort_padded = _padded(n_sort)
        n_neighbors = 8 * max_particle_blocks
        max_nodes = max_blocks * BLOCK_CELLS
        self.n_sort = n_sort
        self.max_blocks = max_blocks
        self.max_particle_blocks = max_particle_blocks

        self.sort_keys = qd.ndarray(qd.u64, shape=(n_sort_padded,))
        self.sort_keys_tmp = qd.ndarray(qd.u64, shape=(n_sort_padded,))
        self.sort_ids = qd.ndarray(qd.i32, shape=(n_sort_padded,))
        self.sort_ids_tmp = qd.ndarray(qd.i32, shape=(n_sort_padded,))
        self.particle_sort = DynamicRadixSort(qd.u64, n_sort_padded)

        self.pblock_keys = qd.ndarray(qd.u64, shape=(max_particle_blocks,))
        self.nbr_keys = qd.ndarray(qd.u64, shape=(n_neighbors,))
        self.nbr_keys_tmp = qd.ndarray(qd.u64, shape=(n_neighbors,))
        self.nbr_perm = qd.ndarray(qd.i32, shape=(n_neighbors,))
        self.nbr_perm_tmp = qd.ndarray(qd.i32, shape=(n_neighbors,))
        self.block_sort = DynamicRadixSort(qd.u64, n_neighbors)
        self.block_flags = qd.ndarray(qd.i32, shape=(n_neighbors,))
        self.block_ranks = qd.ndarray(qd.i32, shape=(n_neighbors,))
        self.block_scan = DynamicExclusiveSum(n_neighbors)
        self.block_coords = qd.ndarray(qd.u64, shape=(max_blocks,))
        self.hash = OpenAddressingHash(max_blocks)

        self.cell_flags = qd.ndarray(qd.i32, shape=(n_sort_padded,))
        self.cell_ranks = qd.ndarray(qd.i32, shape=(n_sort_padded,))
        self.cell_scan = DynamicExclusiveSum(n_sort_padded)
        self.cell_starts = qd.ndarray(qd.i32, shape=(n_sort_padded + 1,))
        self.task_flags = qd.ndarray(qd.i32, shape=(n_sort_padded,))
        self.task_ranks = qd.ndarray(qd.i32, shape=(n_sort_padded,))
        self.task_scan = DynamicExclusiveSum(n_sort_padded)
        self.cell_seg_start = qd.ndarray(qd.i32, shape=(max_nodes,))
        self.cell_seg_end = qd.ndarray(qd.i32, shape=(max_nodes,))

        self.node_mass = qd.ndarray(gs.qd_float, shape=(max_nodes,))
        self.node_fixed = qd.ndarray(qd.i32, shape=(max_nodes,))
        # Per displacement component: held at zero by the Newton solve (a fixed node, or a wall the node moves into)
        self.dof_fixed = qd.ndarray(qd.i32, shape=(3 * max_nodes,))
        self.node_v = qd.ndarray(gs.qd_float, shape=(3 * max_nodes,))
        self.node_v_free = qd.ndarray(gs.qd_float, shape=(3 * max_nodes,))
        self.node_v_solved = qd.ndarray(gs.qd_float, shape=(3 * max_nodes,))

    @property
    def max_nodes(self) -> int:
        return self.max_blocks * BLOCK_CELLS


@qd.data_oriented
class MpmNewton:
    """Newton state of the grid displacement `u`, and the per-particle staging of the elastic assembly."""

    def __init__(self, n_sort: int, max_nodes: int) -> None:
        self.allocate(n_sort, max_nodes)

    def allocate(self, n_sort: int, max_nodes: int) -> None:
        n_partials = _padded(n_sort) // REDUCE_BLOCK_DIM + _padded(max_nodes) // REDUCE_BLOCK_DIM
        self.n_particle_partials = _padded(n_sort) // REDUCE_BLOCK_DIM
        self.u = qd.ndarray(gs.qd_float, shape=(3 * max_nodes,))
        self.u_temp = qd.ndarray(gs.qd_float, shape=(3 * max_nodes,))
        self.du = qd.ndarray(gs.qd_float, shape=(3 * max_nodes,))
        # Per sorted particle: V dt^2 P(F_trial) F^T, the factor of its elastic gradient contribution to every node
        self.g_p = qd.ndarray(gs.qd_float, shape=(9 * _padded(n_sort),))
        self.energy_partials = qd.ndarray(gs.qd_float, shape=(n_partials,))


@qd.data_oriented
class MpmLinearSystem:
    """Block-sparse Newton matrix over 3x3 node blocks, its right-hand side, solution, preconditioner and PCG state."""

    def __init__(self, max_nodes: int, max_triplets: int) -> None:
        self.row_offset = 0
        self.matrix = BCOOMatrix(
            shape=(max_nodes, max_nodes),
            block_shape=(3, 3),
            value_type=gs.qd_float,
            symmetric=True,
            initial_triplets=0,
            max_triplets=max_triplets,
        )
        self.pcg = PCGWorkspace(max_nodes, gs.qd_float)
        self.allocate_rows(max_nodes)

    def allocate_rows(self, max_nodes: int) -> None:
        """(Re)allocate everything sized by the node rows, keeping this object."""
        self.max_nodes = max_nodes
        self.pcg.allocate(max_nodes)
        self.b_rhs = qd.ndarray(gs.qd_float, shape=(self.pcg.r.shape[0],))
        self.solution = qd.ndarray(gs.qd_float, shape=(self.pcg.r.shape[0],))
        # Brick preconditioner: per brick, its packed 48 x 48 matrix and then inverse, and the mask of its nodes
        # coupled to another node of the brick
        n_bricks = max(max_nodes // BRICK_NODES, 1)
        self.brick = qd.ndarray(gs.qd_float, shape=(n_bricks * BRICK_SCALARS,))
        self.brick_mask = qd.ndarray(qd.i32, shape=(n_bricks,))


def get_mpm_newton_control() -> NewtonControl:
    return NewtonControl(gs.qd_float)


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
    """Triplets of one kinetic diagonal per node and STENCIL_BLOCKS per elastic task."""
    return max_blocks * BLOCK_CELLS + STENCIL_BLOCKS * n_tasks


def initial_capacities(positions: np.ndarray, inv_dx: float) -> tuple[int, int]:
    """Block and triplet capacities from the particle positions, (n_envs, n_particles, 3): twice the initial demand.

    The blocks are those of the particles and their positive-octant neighbours; the elastic tasks are bounded by the
    active cells plus one per 32 particles.
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
