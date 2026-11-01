"""Data of the implicit MPM solver.

The particle state keeps the layout of the solver's public fields. This module holds the data the implicit step adds:
the material table, the sparse grid rebuilt every substep, the Newton state over the grid displacement, and the linear
system the Newton solve assembles. The grid, Newton and linear-system roots are mutable data-oriented objects, so that
growing a capacity replaces an array on an existing path while every kernel keeps the same object.
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
# Bricks inverted per thread block of the inversion, one warp each
BRICK_INVERT_WARPS = 4

# Soft contact rows of a particle: one domain-wall row per axis, then the pyramid rows of its deepest Rigid contact
WALL_ROWS = 3
PYRAMID_ROWS = 4
CONTACT_ROWS = WALL_ROWS + PYRAMID_ROWS


class MpmStaticConfig(metaclass=array_class.AutoInitMeta):
    """Compile-time switches of the implicit MPM kernels: the code paths a scene takes, never its sizes or values."""

    # MPM is coupled with Rigid, in one Newton solve
    has_rigid: bool
    # The links info of Rigid is batched over the environments
    rigid_batch_links_info: bool
    # Rigid has constraint rows, which join the solve
    has_rigid_rows: bool
    # The one switch of Rigid's collider configuration that the signed distance field (SDF) queries read. This config
    # stands in for it there, so that the MPM kernels compile independently of the rest of that configuration.
    has_terrain: bool


def _scalar(dtype, value=0):
    result = qd.ndarray(dtype, shape=())
    np_dtype = {qd.i32: np.int32, qd.f32: np.float32, qd.f64: np.float64}[dtype]
    result.from_numpy(np.array(value, dtype=np_dtype))
    return result


@qd.data_oriented
class MpmParams:
    """Scene values of the implicit MPM kernels, as device scalars, so that a scene of another size or resolution runs
    the kernels compiled for the previous one."""

    def __init__(
        self,
        n_envs: int,
        n_particles: int,
        substep_dt: float,
        dx: float,
        particle_volume: float,
        mass_scale: float,
        n_materials: int,
        wall_sol_params,
        wall_lower,
        wall_upper,
    ) -> None:
        self.n_envs = _scalar(qd.i32, n_envs)
        self.n_particles = _scalar(qd.i32, n_particles)
        self.substep_dt = _scalar(gs.qd_float, substep_dt)
        self.dx = _scalar(gs.qd_float, dx)
        self.inv_dx = _scalar(gs.qd_float, 1.0 / dx)
        self.particle_volume = _scalar(gs.qd_float, particle_volume)
        self.mass_scale = _scalar(gs.qd_float, mass_scale)
        self.n_materials = _scalar(qd.i32, n_materials)
        # Start the Newton solve from the inertial prediction u = dt v_free instead of u = 0: contact rows the substep
        # will violate are then active at the first assembly, so that the first direction already sees them. A barrier
        # against penetration needs the start u = 0, free of penetration, for its continuous collision detection, so
        # this stays a switch.
        self.newton_start_inertial = _scalar(qd.i32, 0)
        # Thickness of the FEM surface for the MPM contact rows, in metres: a triangle has no volume of its own
        self.fem_contact_thickness = _scalar(gs.qd_float, 0.5 * dx)
        # Penalty stiffness of an FEM contact row, in units of the row's effective mass, the inverse of its inverse
        # weight: the row energy is 1/2 stiffness m_eff c^2 in the joint dt^2-scaled units
        self.fem_contact_stiffness = _scalar(gs.qd_float, 10.0)
        # The contact rows of MPM particles against PBD particles: the thickness beyond the PBD radius, the penalty
        # stiffness in units of the row's effective mass, and the friction coefficient, which PBD particles do not hold
        self.pbd_contact_thickness = _scalar(gs.qd_float, 0.5 * dx)
        self.pbd_contact_stiffness = _scalar(gs.qd_float, 10.0)
        self.pbd_contact_friction = _scalar(gs.qd_float, 0.15)
        # Rigid's solref / solimp parameters of the wall rows: timeconst, dampratio, dmin, dmax, width, mid, power
        self.wall_sol_params = qd.ndarray(gs.qd_float, shape=(7,))
        self.wall_sol_params.from_numpy(np.array(wall_sol_params, dtype=gs.np_float))
        # Planes of the domain walls, `CubeBoundary.lower` / `upper`
        self.wall_lower = qd.ndarray(gs.qd_float, shape=(3,))
        self.wall_upper = qd.ndarray(gs.qd_float, shape=(3,))
        self.wall_lower.from_numpy(wall_lower)
        self.wall_upper.from_numpy(wall_upper)


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class MpmParticles:
    """Particle state of every frame of the substep window, shape (frames, n_particles, n_envs)."""

    kind: ClassVar[array_class.DataKind] = array_class.DataKind.STATE

    pos: qd.Tensor
    vel: qd.Tensor
    C: qd.Tensor
    F: qd.Tensor
    actu: qd.Tensor
    Jp: qd.Tensor


def get_mpm_particles(n_frames: int, n_particles: int, n_envs: int, requires_grad: bool = False) -> MpmParticles:
    shape = (n_frames, n_particles, n_envs)
    return MpmParticles(
        pos=array_class.V(dtype=gs.qd_vec3, shape=shape, needs_grad=requires_grad),
        vel=array_class.V(dtype=gs.qd_vec3, shape=shape, needs_grad=requires_grad),
        C=array_class.V(dtype=gs.qd_mat3, shape=shape, needs_grad=requires_grad),
        F=array_class.V(dtype=gs.qd_mat3, shape=shape, needs_grad=requires_grad),
        actu=array_class.V(dtype=gs.qd_float, shape=shape, needs_grad=requires_grad),
        Jp=array_class.V(dtype=gs.qd_float, shape=shape, needs_grad=requires_grad),
    )


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class MpmParticlesNG:
    """Particle state without gradient, shape (frames, n_particles, n_envs)."""

    kind: ClassVar[array_class.DataKind] = array_class.DataKind.STATE

    active: qd.Tensor


def get_mpm_particles_ng(n_frames: int, n_particles: int, n_envs: int) -> MpmParticlesNG:
    return MpmParticlesNG(active=array_class.V(dtype=gs.qd_bool, shape=(n_frames, n_particles, n_envs)))


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class MpmParticlesInfo:
    """Static per-particle data, shape (n_particles,)."""

    kind: ClassVar[array_class.DataKind] = array_class.DataKind.INFO

    material_idx: qd.Tensor
    mass: qd.Tensor
    default_Jp: qd.Tensor
    free: qd.Tensor
    muscle_group: qd.Tensor
    muscle_direction: qd.Tensor
    # Inverse weight matrix of the particle in its contact rows, a model constant (see func_contact_invweight)
    contact_invweight: qd.Tensor


def get_mpm_particles_info(n_particles: int) -> MpmParticlesInfo:
    shape = (n_particles,)
    return MpmParticlesInfo(
        material_idx=array_class.V(dtype=gs.qd_int, shape=shape),
        mass=array_class.V(dtype=gs.qd_float, shape=shape),
        default_Jp=array_class.V(dtype=gs.qd_float, shape=shape),
        free=array_class.V(dtype=gs.qd_bool, shape=shape),
        muscle_group=array_class.V(dtype=gs.qd_int, shape=shape),
        muscle_direction=array_class.V(dtype=gs.qd_vec3, shape=shape),
        contact_invweight=array_class.V(dtype=gs.qd_mat3, shape=shape),
    )


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class MpmMaterialTable:
    """Parameter-major material table, `material_table[param * n_materials + slot]`, and the energy family and the
    return map of each slot."""

    kind: ClassVar[array_class.DataKind] = array_class.DataKind.CONSTANT

    material_table: qd.Tensor
    mat_family: qd.Tensor
    mat_plastic: qd.Tensor


def get_mpm_material_table(table: np.ndarray, family: np.ndarray, plastic: np.ndarray) -> MpmMaterialTable:
    material_table = array_class.V(dtype=gs.qd_float, shape=(max(table.size, 1),))
    mat_family = array_class.V(dtype=gs.qd_int, shape=(max(family.size, 1),))
    mat_plastic = array_class.V(dtype=gs.qd_int, shape=(max(plastic.size, 1),))
    material_table.from_numpy(np.resize(table.astype(gs.np_float), max(table.size, 1)))
    mat_family.from_numpy(np.resize(family.astype(gs.np_int), max(family.size, 1)))
    mat_plastic.from_numpy(np.resize(plastic.astype(gs.np_int), max(plastic.size, 1)))
    return MpmMaterialTable(material_table=material_table, mat_family=mat_family, mat_plastic=mat_plastic)


def _padded(length: int, multiple: int = REDUCE_BLOCK_DIM) -> int:
    return max(((max(length, 1) + multiple - 1) // multiple) * multiple, multiple)


@qd.data_oriented
class MpmGrid:
    """Block-sparse grid rebuilt from the particles every substep.

    Particles are sorted by (environment, block, cell). Every particle block activates its positive-octant neighbour
    blocks as node storage. A block slot is the rank of its key among the sorted unique keys, and a node or cell slot is
    `block_slot * BLOCK_CELLS + local index`.

    `errno` holds the error flags of the MPM solver per environment, which the kernels building and reading the grid
    raise instead of indexing out of bounds (see mpm_func.flag_env_error).
    """

    def __init__(self, n_sort: int, max_blocks: int, max_particle_blocks: int, errno) -> None:
        self.errno = errno
        self.allocate(n_sort, max_blocks, max_particle_blocks)
        # Number of particle keys sorted, n_envs * n_particles
        self.n_sort_keys = _scalar(qd.i32)
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
        # Per displacement component: held at zero by the Newton solve
        self.dof_fixed = qd.ndarray(qd.i32, shape=(3 * max_nodes,))
        # Diagonal 3x3 block of every node of the physical Hessian (kinetic and elastic), from which the inverse weights
        # of the particles in their contact rows are computed once (see func_contact_invweight)
        self.node_diag = qd.ndarray(gs.qd_float, shape=(9 * max_nodes,))
        self.node_v = qd.ndarray(gs.qd_float, shape=(3 * max_nodes,))
        self.node_v_free = qd.ndarray(gs.qd_float, shape=(3 * max_nodes,))
        self.node_v_solved = qd.ndarray(gs.qd_float, shape=(3 * max_nodes,))

    @property
    def max_nodes(self) -> int:
        return self.max_blocks * BLOCK_CELLS


@qd.data_oriented
class MpmNewton:
    """Newton state of the grid displacement `u`, the per-particle staging of the assembly, and the soft contact rows
    of the particles."""

    def __init__(self, n_sort: int, max_nodes: int) -> None:
        self.allocate(n_sort, max_nodes)

    def allocate(self, n_sort: int, max_nodes: int) -> None:
        n_padded = _padded(n_sort)
        n_partials = n_padded // REDUCE_BLOCK_DIM + _padded(max_nodes) // REDUCE_BLOCK_DIM
        # Offset of the node partials of the energy, after the particle partials
        self.n_particle_partials = _scalar(qd.i32, n_padded // REDUCE_BLOCK_DIM)
        self.u = qd.ndarray(gs.qd_float, shape=(3 * max_nodes,))
        self.u_temp = qd.ndarray(gs.qd_float, shape=(3 * max_nodes,))
        self.du = qd.ndarray(gs.qd_float, shape=(3 * max_nodes,))
        # Adjoint of the node momentum and mass of the particle-to-grid transfer, which the backward pass of a substep
        # gathers to the particles
        self.adj_mom = qd.ndarray(gs.qd_float, shape=(3 * max_nodes,))
        self.adj_mass = qd.ndarray(gs.qd_float, shape=(max_nodes,))
        # Per sorted particle: V dt^2 P(F_trial) F^T, the factor of its elastic gradient contribution to every node
        self.g_p = qd.ndarray(gs.qd_float, shape=(9 * n_padded,))
        # Per sorted particle: the gradient of its contact rows with respect to its trial displacement sum_a w_a u_a
        self.g_contact = qd.ndarray(gs.qd_float, shape=(3 * n_padded,))
        # Per sorted particle: the gradient and Hessian of its Rigid contact rows alone, which couple it with the dofs
        # of its link, and the displacement of its contact point on the link over the substep
        self.g_rigid = qd.ndarray(gs.qd_float, shape=(3 * n_padded,))
        self.h_rigid = qd.ndarray(gs.qd_float, shape=(9 * n_padded,))
        self.contact_dr = qd.ndarray(gs.qd_float, shape=(3 * n_padded,))
        # Per sorted particle: the spring that attaches it to its target (see func_build_constraint_rows), its stiffness
        # dt^2 k and damping dt c, both zero for a free particle, and the target. They start at zero, since the substeps
        # build them only once a constraint exists.
        self.constraint_spring = qd.ndarray(gs.qd_float, shape=(n_padded,))
        self.constraint_damping = qd.ndarray(gs.qd_float, shape=(n_padded,))
        self.constraint_target = qd.ndarray(gs.qd_float, shape=(3 * n_padded,))
        for array in (self.constraint_spring, self.constraint_damping, self.constraint_target):
            array.from_numpy(np.zeros(array.shape, dtype=gs.np_float))
        self.energy_partials = qd.ndarray(gs.qd_float, shape=(n_partials,))
        # Soft contact rows, CONTACT_ROWS per sorted particle on its trial displacement dx = sum_a w_a u_a: direction J,
        # stiffness D (zero where there is no row) and target t of the row term 1/2 D min(0, J.dx - t)^2, and the
        # Rigid link of the particle's pyramid rows (-1 if none)
        self.contact_dir = qd.ndarray(gs.qd_float, shape=(3 * CONTACT_ROWS * n_padded,))
        self.contact_D = qd.ndarray(gs.qd_float, shape=(CONTACT_ROWS * n_padded,))
        self.contact_t = qd.ndarray(gs.qd_float, shape=(CONTACT_ROWS * n_padded,))
        self.contact_link = qd.ndarray(qd.i32, shape=(n_padded,))
        # The FEM surface triangle of the particle's pyramid rows (-1 if none), and the barycentric coordinates of its
        # closest point on it
        self.contact_tri = qd.ndarray(qd.i32, shape=(n_padded,))
        self.contact_bary = qd.ndarray(gs.qd_float, shape=(3 * n_padded,))
        # The PBD particle of the particle's pyramid rows, in PBD's reordered layout (-1 if none)
        self.contact_pbd = qd.ndarray(qd.i32, shape=(n_padded,))
        # Nearest obstacle of each sorted particle during a bounding volume hierarchy query: its gap, its index, and,
        # for a triangle, the distance to it and the alignment of its normal with the direction to the particle (which
        # together pick the triangle), the barycentric coordinates of the closest point and the contact normal. The
        # query point and the frame at which the obstacles are read sit alongside, for the filter measuring the gaps.
        self.query_phi = qd.ndarray(gs.qd_float, shape=(n_padded,))
        self.query_idx = qd.ndarray(qd.i32, shape=(n_padded,))
        self.query_dist = qd.ndarray(gs.qd_float, shape=(n_padded,))
        self.query_align = qd.ndarray(gs.qd_float, shape=(n_padded,))
        self.query_bary = qd.ndarray(gs.qd_float, shape=(3 * n_padded,))
        self.query_normal = qd.ndarray(gs.qd_float, shape=(3 * n_padded,))
        self.query_pos = qd.ndarray(gs.qd_float, shape=(3 * n_padded,))
        self.query_frame = qd.ndarray(qd.i32, shape=())


@qd.data_oriented
class MpmLinearSystem:
    """Block-sparse Newton matrix over 3x3 node blocks, its right-hand side, solution, preconditioner and conjugate
    gradient state."""

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
        # The serial inversion of a CPU backend: per brick, the equilibration of its rows and the pivot column
        n_scratch = n_bricks * 2 * BRICK_DIM if gs.backend == gs.cpu else 1
        self.brick_scratch = qd.ndarray(gs.qd_float, shape=(n_scratch,))


# Links in contact with MPM per environment whose contact stiffness the Rigid preconditioner takes in. The stiffness on
# any further link is left to the conjugate gradient iterations.
MAX_PRECONDITIONED_LINKS = 16


@qd.data_oriented
class MpmRigidCoupling:
    """Rigid unknowns of the joint Newton solve of MPM and Rigid, and the Rigid part of its conjugate gradient vectors.

    The Rigid unknown is x = dt^2 qacc per dof and environment, with Rigid's own cost scaled by dt^4 to the displacement
    units of the MPM energy. `qacc`, `Ma` and `cost` hold Rigid's evaluation of x. `r`, `z`, `p`, `ap` and `sol` are the
    Rigid rows of the conjugate gradient residual, preconditioned residual, direction, operator image and solution. Per
    link, `link_V` holds the spatial displacement of a dof vector and `link_F` accumulates a spatial force, linear then
    angular about the root center of mass.
    """

    def __init__(self, n_dofs: int, n_links: int, n_envs: int) -> None:
        shape = (max(n_dofs, 1), n_envs)
        self.x = qd.ndarray(gs.qd_float, shape=shape)
        self.x_temp = qd.ndarray(gs.qd_float, shape=shape)
        self.dx = qd.ndarray(gs.qd_float, shape=shape)
        self.grad = qd.ndarray(gs.qd_float, shape=shape)
        self.qacc = qd.ndarray(gs.qd_float, shape=shape)
        self.Ma = qd.ndarray(gs.qd_float, shape=shape)
        self.r = qd.ndarray(gs.qd_float, shape=shape)
        self.z = qd.ndarray(gs.qd_float, shape=shape)
        self.p = qd.ndarray(gs.qd_float, shape=shape)
        self.ap = qd.ndarray(gs.qd_float, shape=shape)
        self.sol = qd.ndarray(gs.qd_float, shape=shape)
        self.cost = qd.ndarray(gs.qd_float, shape=(n_envs,))
        link_shape = (max(n_links, 1), n_envs)
        self.link_F = qd.ndarray(gs.qd_float, shape=(*link_shape, 6))
        self.link_V = qd.ndarray(gs.qd_float, shape=(*link_shape, 6))
        # The MPM contact stiffness on the links for the preconditioner, by the Woodbury identity on Rigid's factor A.
        # Per link, its 6 x 6 spatial stiffness K_l, row-major. Per environment, the links in contact (at most
        # `n_contact_links`), the cdof map U of their chains with one column per spatial component, W = A^-1 U, and the
        # lower-upper (LU) factor of S = I + K U^T W with its row pivots. `solve_rhs` / `solve_out` stage one column of
        # a solve.
        self.n_contact_links = min(max(n_links, 1), MAX_PRECONDITIONED_LINKS)
        n_cols = 6 * self.n_contact_links
        self.link_K = qd.ndarray(gs.qd_float, shape=(*link_shape, 36))
        self.contact_links = qd.ndarray(qd.i32, shape=(n_envs, self.n_contact_links))
        self.n_links_in_contact = qd.ndarray(qd.i32, shape=(n_envs,))
        self.wb_U = qd.ndarray(gs.qd_float, shape=(max(n_dofs, 1), n_envs, n_cols))
        self.wb_W = qd.ndarray(gs.qd_float, shape=(max(n_dofs, 1), n_envs, n_cols))
        self.wb_S = qd.ndarray(gs.qd_float, shape=(n_envs, n_cols, n_cols))
        self.wb_pivot = qd.ndarray(qd.i32, shape=(n_envs, n_cols))
        self.wb_y = qd.ndarray(gs.qd_float, shape=(n_envs, n_cols))
        self.solve_rhs = qd.ndarray(gs.qd_float, shape=shape)
        self.solve_out = qd.ndarray(gs.qd_float, shape=shape)


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
