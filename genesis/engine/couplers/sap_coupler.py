import dataclasses
import math
import textwrap
from collections.abc import Iterator
from enum import auto
from typing import TYPE_CHECKING

import numpy as np

import igl

import quadrants as qd
from quadrants.algorithms import sort as radix_sort
from quadrants.algorithms import sort_scratch_slots

import genesis as gs
import genesis.utils.element as eu
import genesis.utils.geom as gu
import genesis.utils.simt as su
from genesis.constants import IntEnum
from genesis.engine.bvh import (
    BVHLeaves,
    BVHQueryResults,
    BVHQueryState,
    build_bvh,
    func_bvh_query_leaves,
    func_no_filter,
    get_bvh_data,
)
from genesis.engine.solvers.base_solver import CheckpointMixin, ErrnoMixin, ErrorCode
from genesis.engine.solvers.fem_solver import (
    ELEMENTS,
    VERTICES,
    FEMInfo,
    FEMState,
    FEMVertexConstraints,
    func_elastic_product,
    func_element_mapping_matrix,
)
from genesis.engine.solvers.rigid.abd.forward_dynamics import func_solve_mass_block
from genesis.engine.solvers.rigid.abd.forward_kinematics import kernel_update_all_verts
from genesis.engine.solvers.rigid.data import (
    DOFS,
    GEOMS,
    LINKS,
    DofsState,
    EntitiesInfo,
    EqualitiesInfo,
    FacesInfo,
    GeomsInfo,
    GeomsState,
    JointsInfo,
    LinksInfo,
    LinksState,
    RigidInfo,
    VertsInfo,
    VertsState,
)
from genesis.options.solvers import SAPCouplerOptions
from genesis.repr_base import RBC
from genesis.utils.data import (
    ENVS,
    Config,
    Data,
    DataItem,
    DataKind,
    Size,
    eoa,
    field,
    iter_data,
)
from genesis.utils.misc import qd_to_numpy, tensor_to_array

if TYPE_CHECKING:
    from genesis.engine.simulator import Simulator


class SAPErrorCode(ErrorCode):
    """The error flags the SAP coupler writes in its errno (see 'ErrnoMixin.register_errors')."""

    OVERFLOW_CANDIDATE_CONTACTS = auto()
    OVERFLOW_CONTACTS = auto()
    INVALID_SOLVE = auto()


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class SAPContactQueriesState:
    """The box queries of the SAP coupler, one per contact handler traversing a tree (see SAPCoupler._init_bvh)."""

    fem_self: BVHQueryState
    rigid_tri: BVHQueryState
    rigid_tet: BVHQueryState


MARCHING_TETS_EDGE_TABLE = (
    (-1, -1, -1, -1),
    (0, 3, 2, -1),
    (0, 1, 4, -1),
    (4, 3, 2, 1),
    (1, 2, 5, -1),
    (0, 3, 5, 1),
    (0, 2, 5, 4),
    (3, 5, 4, -1),
    (3, 4, 5, -1),
    (4, 5, 2, 0),
    (1, 5, 3, 0),
    (1, 5, 2, -1),
    (1, 2, 3, 4),
    (0, 4, 1, -1),
    (0, 2, 3, -1),
    (-1, -1, -1, -1),
)

TET_EDGES = (
    (0, 1),
    (1, 2),
    (2, 0),
    (0, 3),
    (1, 3),
    (2, 3),
)

# Slope of the cost along a Newton step below which the line search takes the full step, at the round-off level of the
# cost: the relative bound of double precision, scaled to the precision of the simulation. A looser bound takes full
# steps that increase the cost, and the Newton iterations cycle.
LINESEARCH_COST_ATOL = 1e-30
LINESEARCH_COST_RTOL = 1e-15

# Smallest component of the pressure gradient of a tet along the normal of its contact, towards the inside of its body,
# relative to the norm of that gradient and in units of the precision of the simulation, below which the component is
# round-off. The stiffness of a contact is its area times this component, harmonically averaged over both sides, so
# that a contact without a pressure increasing into a side has no stiffness.
CONTACT_GRADIENT_RTOL = 4.0

# Smallest area of a contact polygon, stiffness of a contact and largest signed distance of a contact, in SI units,
# below which (above which for the distance) a contact transmits no force a solve resolves. These are physical floors,
# the same in both precisions, whereas the round-off bounds above scale with the precision. In single precision, the
# separation floor lies below the resolution of a position, so that it reads as any separation at all.
CONTACT_AREA_MIN = 1e-10
CONTACT_STIFFNESS_MIN = 1e-10
CONTACT_SEPARATION_MAX = 1e-10

# Cosine of the largest angle between the pressure gradient of a tet and the normal of a contact plane of FEM tets,
# beyond which the pair has no contact. The bound lies past a right angle, so that it drops only the pairs whose normal
# points against the pressure increase into a side.
COS_ANGLE_THRESHOLD = math.cos(math.pi * 5.0 / 8.0)

# Largest residual of a SAP solve stopped at its Newton iteration cap, relative to the momentum and the impulses of its
# environment (see func_update_envs_active): a momentum left unbalanced above a tenth of them leaves velocities off by
# as much, which inject energy into the contacts, while a solve stopped below it, as on a rare spike of the iterations,
# keeps a meaningful output.
CAP_RESIDUAL_MAX = 0.1

# Default tolerance of a SAP solve on its gradient relative to the momentum of its environment, by precision: double
# precision solves for accuracy, single precision for speed (see SAPCouplerOptions).
CONVERGENCE_RTOL_DEFAULT = {np.float64: 1e-4, np.float32: 1e-3}

# The items of a chunk, the unit of work whose sums every per-environment total adds up in a fixed order, so that the
# solve is reproducible bitwise: the lanes of a GPU subgroup take one item each and sum them along a fixed tree, a CPU
# thread takes them in a row. The vertices, dofs, elements and constraints of each environment split into chunks.
CHUNK_SIZE = 32
# The sums a chunk of FEM vertices or rigid dofs holds, and a chunk of constraints (see VertexDofSum, ConstraintSum).
N_VERTEX_DOF_SUMS = 8
N_CONSTRAINT_SUMS = 3
# The shares of the contacts of a rigid link that the subgroups of a link sum apart on GPU, so that a link with many
# contacts takes several subgroups, and that each dof adds up in order (see func_compute_link_wrench).
N_LINK_WRENCH_SHARES = 2
# The most dofs of the kinematic chain of a contact the kernels hold in registers, by scene (see SAPConfig)
CHAIN_DOFS_SIZES = (8, 16, 32, 64)


class ConstraintOp(IntEnum):
    """Enum for the operations of the SAP solve on its constraints, the contact pairs and the equality constraints."""

    GRADIENT_HESSIAN = auto()
    """Impulse and Hessian of each constraint at the current velocities."""
    MATRIX_VECTOR_PRODUCT = auto()
    """Product of the Hessian of each constraint with the Jacobian of the search direction."""
    SEARCH_DIRECTION = auto()
    """Change of the constraint velocities along the Newton step."""
    LINE_DERIVATIVES = auto()
    """First and second derivatives of the cost along the Newton step."""
    ENERGY = auto()
    """Cost of the constraints at the start of the line search."""


class VertexDofOp(IntEnum):
    """Enum for the operations of the SAP solve on the FEM vertices and the rigid dofs."""

    GRADIENT = auto()
    """Gradient with the impulses of the constraints, preconditioner, and the norms of convergence."""
    PCG_INIT = auto()
    """Residual and preconditioned residual at the start of the conjugate gradient, and their product."""
    PCG_SEARCH_DIRECTION = auto()
    """Search direction, and the product of the mass matrix with it."""
    PCG_PTAP = auto()
    """Product of the Hessian with the search direction, and of the search direction with it."""
    PCG_UPDATE = auto()
    """Solution, residual and preconditioned residual stepped along the search direction."""
    LINESEARCH_INIT = auto()
    """Velocities at the start of the line search, and the slope of the cost along the Newton step."""
    ENERGY = auto()
    """Free motion part of the cost at the start of the line search."""
    LINESEARCH_D2ELLA = auto()
    """Second derivative of the free motion part of the cost along the Newton step."""
    LINESEARCH_VELOCITY = auto()
    """Velocities at the current step size, and the free motion part of the slope of the cost."""


class VertexDofSum(IntEnum):
    """
    The sums of the chunks of vertices and dofs (see func_compute_vertices_dofs).

    Each operation writes its own, which the operations that read them consume before another operation writes them
    again: the norms of convergence, then r^T z and the residual restricted to the rigid motions of the entity of the
    chunk, p^T A p, then the slope, energy and second derivative of the free motion part of the cost along the Newton
    step at its start, and its first derivative.
    """

    GRADIENT_NORM = 0
    MOMENTUM_NORM = 1
    IMPULSE_NORM = 2
    RTZ = 0
    COARSE_RHS = 1
    PTAP = 7
    SLOPE = 0
    ENERGY = 1
    D2ELLA = 2
    DELL = 3

    @staticmethod
    def of_op(op: VertexDofOp) -> tuple[int, ...]:
        """Return the sums an operation writes (see VertexDofOp)."""
        coarse_rhs = tuple(range(VertexDofSum.COARSE_RHS, VertexDofSum.COARSE_RHS + 6))
        match op:
            case VertexDofOp.GRADIENT:
                return (VertexDofSum.GRADIENT_NORM, VertexDofSum.MOMENTUM_NORM, VertexDofSum.IMPULSE_NORM)
            case VertexDofOp.PCG_INIT | VertexDofOp.PCG_UPDATE:
                return (VertexDofSum.RTZ, *coarse_rhs)
            case VertexDofOp.PCG_SEARCH_DIRECTION:
                return ()
            case VertexDofOp.PCG_PTAP:
                return (VertexDofSum.PTAP,)
            case VertexDofOp.LINESEARCH_INIT:
                return (VertexDofSum.SLOPE,)
            case VertexDofOp.ENERGY:
                return (VertexDofSum.ENERGY,)
            case VertexDofOp.LINESEARCH_D2ELLA:
                return (VertexDofSum.D2ELLA,)
            case VertexDofOp.LINESEARCH_VELOCITY:
                return (VertexDofSum.DELL,)


class ConstraintSum(IntEnum):
    """
    The sums of the chunks of constraints (see func_compute_constraints): the first and second derivatives of their cost
    along the Newton step, and their cost at its start.
    """

    DELL = 0
    D2ELL = 1
    ENERGY = 2

    @staticmethod
    def of_op(op: ConstraintOp) -> tuple[int, ...]:
        """Return the sums an operation writes (see ConstraintOp)."""
        match op:
            case ConstraintOp.LINE_DERIVATIVES:
                return (ConstraintSum.DELL, ConstraintSum.D2ELL)
            case ConstraintOp.ENERGY:
                return (ConstraintSum.ENERGY,)
            case ConstraintOp.GRADIENT_HESSIAN | ConstraintOp.MATRIX_VECTOR_PRODUCT | ConstraintOp.SEARCH_DIRECTION:
                return ()


# The vertices and elements of the tetrahedral volume meshes of the rigid geoms, and the rigid triangles touching FEM
RIGID_VOLUME_VERTS = Size("rigid_volume_verts")
RIGID_VOLUME_ELEMS = Size("rigid_volume_elems")
RIGID_TRI_FACES = Size("rigid_tri_faces")
# The FEM entities, the chunks of the FEM vertices then of the rigid dofs, and of the FEM elements (see CHUNK_SIZE)
FEM_ENTITIES = Size("fem_entities")
VERTEX_CHUNKS = Size("vertex_chunks")
VERTEX_DOF_CHUNKS = Size("vertex_dof_chunks")
ELEMENT_CHUNKS = Size("element_chunks")
# The constraint handlers, the slots of their constraints over all environments, the chunks of the sorted constraints of
# every environment, and the scratch of the device radix sort of their keys.
CONSTRAINT_HANDLERS = Size("constraint_handlers")
CONSTRAINT_SLOTS = Size("constraint_slots")
CONSTRAINT_UNITS = Size("constraint_units")
CONSTRAINT_SORT_SLOTS = Size("constraint_sort_slots")
# The entries of the lists of the contacts of the FEM vertices and of the rigid links, and the wrench shares of a link
FEM_INCIDENCE = Size("fem_incidence")
LINK_INCIDENCE = Size("link_incidence")
LINK_WRENCH_SHARES = Size("link_wrench_shares")

# Labels of the arrays that the solvers and the contact types of a scene leave unused
FEM = "fem"
RIGID = "rigid"
FEM_PRESSURE = "fem_pressure"
RIGID_VOLUME = "rigid_volume"
RIGID_RIGID = "rigid_rigid"
RIGID_FEM = "rigid_fem"
TET_TABLES = "tet_tables"
RIGID_CONTACTS = "rigid_contacts"
HOST_SORT = "host_sort"
DEVICE_SORT = "device_sort"


class SAPFEMVertices(Data, kind=DataKind.SCRATCH, shape=(VERTICES, ENVS)):
    """The velocities of the FEM vertices that the SAP solve iterates on, with the vectors of its solves."""

    v: qd.Tensor = field((float, 3))
    # Difference of the velocity with the reference one (see kernel_init_sap_solve)
    v_diff: qd.Tensor = field((float, 3))
    gradient: qd.Tensor = field((float, 3))
    impulse: qd.Tensor = field((float, 3))
    # Diagonal 3x3 block of the Hessian, whose inverse `prec` is the preconditioner
    diag3x3: qd.Tensor = field((float, (3, 3)))
    prec: qd.Tensor = field((float, (3, 3)))
    # The vectors of the preconditioned conjugate gradient (PCG) solve
    x: qd.Tensor = field((float, 3))
    r: qd.Tensor = field((float, 3))
    z: qd.Tensor = field((float, 3))
    p: qd.Tensor = field((float, 3))
    Ap: qd.Tensor = field((float, 3))
    # Velocity the line search starts from
    x_prev: qd.Tensor = field((float, 3))
    # Product of the Hessian of the free motion with the Newton step
    dp: qd.Tensor = field((float, 3))


class SAPRigidDofs(Data, kind=DataKind.SCRATCH, shape=(DOFS, ENVS)):
    """The velocities of the rigid dofs that the SAP solve iterates on, with the vectors of its solves."""

    v: qd.Tensor = field(float)
    # Difference of the velocity with the free-motion one, and its product with the mass matrix
    v_diff: qd.Tensor = field(float)
    mass_v_diff: qd.Tensor = field(float)
    gradient: qd.Tensor = field(float)
    impulse: qd.Tensor = field(float)
    # The vectors of the preconditioned conjugate gradient (PCG) solve, `Mp` the product of the mass matrix with `p`
    x: qd.Tensor = field(float)
    r: qd.Tensor = field(float)
    z: qd.Tensor = field(float)
    p: qd.Tensor = field(float)
    Mp: qd.Tensor = field(float)
    Ap: qd.Tensor = field(float)
    # Velocity the line search starts from
    x_prev: qd.Tensor = field(float)
    # Product of the mass matrix with the Newton step
    dp: qd.Tensor = field(float)


class SAPEnvs(Data, kind=DataKind.SCRATCH, shape=(ENVS,)):
    """The convergence state of the SAP solve of each environment, with the scalars of its solves."""

    is_active: qd.Tensor = field(bool)
    is_pcg_active: qd.Tensor = field(bool)
    is_linesearch_active: qd.Tensor = field(bool)
    # Residual at the last convergence check, relative to the momentum and the impulses
    relative_residual: qd.Tensor = field(float)
    # The scalars of the preconditioned conjugate gradient (PCG) solve
    rTz: qd.Tensor = field(float)
    rTz_fem_threshold: qd.Tensor = field(float)
    rTz_rigid_threshold: qd.Tensor = field(float)
    beta: qd.Tensor = field(float)
    # Cost at the start of the line search and its scale, then at the start of that of the previous Newton iteration
    prev_energy: qd.Tensor = field(float)
    energy_newton_prev: qd.Tensor = field(float)
    # Step size, and slope of the cost along the Newton step
    step_size: qd.Tensor = field(float)
    m: qd.Tensor = field(float)
    # First and second derivatives of the cost along the step, the latter of its free motion alone too
    dell_dalpha: qd.Tensor = field(float)
    d2ellA_dalpha2: qd.Tensor = field(float)
    d2ell_dalpha2: qd.Tensor = field(float)
    dell_scale: qd.Tensor = field(float)
    # Bracket of the step size and its tolerance
    alpha_min: qd.Tensor = field(float)
    alpha_max: qd.Tensor = field(float)
    alpha_tol: qd.Tensor = field(float)
    # Scaled derivative of the cost at the bounds of the bracket and at the step, and the last two bracket widths
    f_lower: qd.Tensor = field(float)
    f_upper: qd.Tensor = field(float)
    f: qd.Tensor = field(float)
    df: qd.Tensor = field(float)
    minus_dalpha: qd.Tensor = field(float)
    minus_dalpha_prev: qd.Tensor = field(float)


class SAPFEMCoarse(Data, kind=DataKind.SCRATCH, shape=(FEM_ENTITIES, ENVS)):
    """The rigid motions of each FEM entity, the coarse space that solves at once what a stiff contact couples."""

    # Point the rotations of the entity are about
    x_ref: qd.Tensor = field((float, 3))
    # Moments of order 0, 1 and 2 of the inertia of the vertices about `x_ref`
    mass: qd.Tensor = field(float)
    mass_first: qd.Tensor = field((float, 3))
    mass_second: qd.Tensor = field((float, (3, 3)))
    # Inverse of the Hessian of the cost restricted to the rigid motions
    A_inv: qd.Tensor = field((float, (6, 6)))
    # Rigid motion solving the residual restricted to the rigid motions
    sol: qd.Tensor = field((float, 6))


class SAPConstraintOrder(Data, kind=DataKind.SCRATCH):
    """The sorted constraints of every environment, their chunks, and the contacts of each vertex and link."""

    # Start of the constraints of each handler in their concatenation, then their total
    handlers_n_constraints: qd.Tensor = field(int, shape=(CONSTRAINT_HANDLERS + 1,))
    # Sort key of each constraint, and the slot of each sorted one in `order` (see kernel_order_constraints)
    key: qd.Tensor = field(qd.u64, shape=(CONSTRAINT_SLOTS,))
    key_scratch: qd.Tensor = field(qd.u64, shape=(CONSTRAINT_SLOTS,))
    order: qd.Tensor = field(int, shape=(CONSTRAINT_SLOTS,))
    order_scratch: qd.Tensor = field(int, shape=(CONSTRAINT_SLOTS,))
    # Scratch of the device radix sort, then of the host one
    sort_scratch: qd.Tensor = field(qd.u32, shape=(CONSTRAINT_SORT_SLOTS,), labels=DEVICE_SORT)
    sort_histogram: qd.Tensor = field(int, shape=(256, ENVS), labels=HOST_SORT)
    envs_cursor: qd.Tensor = field(int, shape=(ENVS,), labels=HOST_SORT)
    n_sorted: qd.Tensor = field(int, shape=())
    # Start of the sorted constraints of each environment
    envs_constraint_start: qd.Tensor = field(int, shape=(ENVS + 1,))
    # Environment and start of each chunk of CHUNK_SIZE sorted constraints, and those of each environment
    units_env: qd.Tensor = field(int, shape=(CONSTRAINT_UNITS,))
    units_start: qd.Tensor = field(int, shape=(CONSTRAINT_UNITS,))
    n_units: qd.Tensor = field(int, shape=())
    envs_unit_start: qd.Tensor = field(int, shape=(ENVS + 1,))
    # Start of the list of the contacts of each FEM vertex, then of each rigid link
    fem_incidence_start: qd.Tensor = field(int, shape=(VERTICES + 1, ENVS), labels=FEM)
    fem_incidence: qd.Tensor = field(int, shape=(FEM_INCIDENCE,), labels=FEM)
    link_incidence_start: qd.Tensor = field(int, shape=(LINKS + 1, ENVS), labels=RIGID_CONTACTS)
    link_incidence: qd.Tensor = field(int, shape=(LINK_INCIDENCE,), labels=RIGID_CONTACTS)
    # Wrench of each share of the contacts of each link (see func_compute_link_wrench)
    links_wrench_force: qd.Tensor = field((float, 3), shape=(LINK_WRENCH_SHARES, ENVS), labels=RIGID_CONTACTS)
    links_wrench_torque: qd.Tensor = field((float, 3), shape=(LINK_WRENCH_SHARES, ENVS), labels=RIGID_CONTACTS)


class SAPSolveLoops(Data, kind=DataKind.SCRATCH, shape=()):
    """The iterations left to each loop of the SAP solve, which runs on the device (see kernel_sap_solve)."""

    # Iterations left past the current one, which the counter takes back once an environment iterates on
    n_newton_next: qd.Tensor = field(int)
    n_pcg_next: qd.Tensor = field(int)
    n_linesearch_next: qd.Tensor = field(int)
    # Ndarrays on every backend, since graph.do_while takes the same ndarray on every call
    newton_counter: qd.Tensor = field(qd.i32, ndarray=True)
    pcg_counter: qd.Tensor = field(qd.i32, ndarray=True)
    linesearch_counter: qd.Tensor = field(qd.i32, ndarray=True)


class SAPState(Data, kind=DataKind.SCRATCH):
    """The data of the SAP coupler that a substep writes: the iterates of its solve and the posed pressure fields."""

    fem_vertices: SAPFEMVertices = field(labels=FEM)
    rigid_dofs: SAPRigidDofs = field(labels=RIGID)
    envs: SAPEnvs
    fem_coarse: SAPFEMCoarse = field(labels=FEM)
    constraint_order: SAPConstraintOrder
    loops: SAPSolveLoops
    # Sums of each chunk of the vertices then of the dofs, of the elements, and of the constraints, added up in order
    vertices_dofs_partials: qd.Tensor = field(float, shape=(VERTEX_DOF_CHUNKS, N_VERTEX_DOF_SUMS, ENVS))
    elements_partials: qd.Tensor = field(float, shape=(ELEMENT_CHUNKS, ENVS), labels=FEM)
    constraints_partials: qd.Tensor = field(float, shape=(CONSTRAINT_UNITS, N_CONSTRAINT_SUMS))
    # Force of each element at each corner, by vertex (see func_compute_fem_corner_force)
    fem_corner_forces: qd.Tensor = field((float, 3), shape=(4 * ELEMENTS, ENVS), labels=FEM)
    fem_pressure_gradient: qd.Tensor = field(
        (float, 3), shape=(ELEMENTS, ENVS), kind=DataKind.DERIVED, labels=FEM_PRESSURE
    )
    rigid_volume_verts: qd.Tensor = field(
        (float, 3), shape=(RIGID_VOLUME_VERTS, ENVS), kind=DataKind.DERIVED, labels=RIGID_VOLUME
    )
    rigid_pressure_gradient: qd.Tensor = field(
        (float, 3), shape=(RIGID_VOLUME_ELEMS, ENVS), kind=DataKind.DERIVED, labels=RIGID_VOLUME
    )


class SAPInfo(Data, kind=DataKind.CONSTANT):
    """The model of the SAP coupler: hydroelastic pressure fields, rigid volume meshes, solve layout and constants."""

    fem_pressure: qd.Tensor = field(float, shape=(VERTICES,), labels=FEM_PRESSURE)
    rigid_volume_verts_rest: qd.Tensor = field((float, 3), shape=(RIGID_VOLUME_VERTS,), labels=RIGID_VOLUME)
    rigid_volume_verts_geom_idx: qd.Tensor = field(int, shape=(RIGID_VOLUME_VERTS,), labels=RIGID_VOLUME)
    rigid_pressure_field: qd.Tensor = field(float, shape=(RIGID_VOLUME_VERTS,), labels=RIGID_VOLUME)
    rigid_volume_elems: qd.Tensor = field((int, 4), shape=(RIGID_VOLUME_ELEMS,), labels=RIGID_VOLUME)
    rigid_volume_elems_geom_idx: qd.Tensor = field(int, shape=(RIGID_VOLUME_ELEMS,), labels=RIGID_VOLUME)
    rigid_pressure_gradient_rest: qd.Tensor = field((float, 3), shape=(RIGID_VOLUME_ELEMS,), labels=RIGID_VOLUME)
    # The collision pairs of the rigid collider at build (see collision_pair_idx in ColliderInfo)
    rigid_collision_pair_idx: qd.Tensor = field(int, shape=(GEOMS, GEOMS), labels=RIGID_RIGID)
    rigid_tri_faces_idx: qd.Tensor = field(int, shape=(RIGID_TRI_FACES,), labels=RIGID_FEM)
    # The edges of a tet cut by a plane, by the code of the vertices above it, and the vertices of the edges of a tet
    marching_tets_edge_table: qd.Tensor = field((int, 4), shape=(len(MARCHING_TETS_EDGE_TABLE),), labels=TET_TABLES)
    tet_edges: qd.Tensor = field((int, 2), shape=(len(TET_EDGES),), labels=TET_TABLES)
    # Range of each chunk of the FEM vertices, and the first chunk of each entity
    chunks_vertex_start: qd.Tensor = field(int, shape=(VERTEX_CHUNKS,), labels=FEM)
    chunks_vertex_end: qd.Tensor = field(int, shape=(VERTEX_CHUNKS,), labels=FEM)
    entities_chunk_start: qd.Tensor = field(int, shape=(FEM_ENTITIES + 1,), labels=FEM)
    # Entity of each FEM vertex, and rank of each element corner in the corners of the vertices
    fem_vertices_entity_idx: qd.Tensor = field(int, shape=(VERTICES,), labels=FEM)
    fem_corners_rank: qd.Tensor = field(int, shape=(4 * ELEMENTS,), labels=FEM)
    # End of the subtree of each link in depth-first order, and the link and tree root of each dof
    links_subtree_end: qd.Tensor = field(int, shape=(LINKS,), labels=RIGID_CONTACTS)
    dofs_link_idx: qd.Tensor = field(int, shape=(DOFS,), labels=RIGID_CONTACTS)
    dofs_root_link_idx: qd.Tensor = field(int, shape=(DOFS,), labels=RIGID_CONTACTS)
    # First constraint slot of each handler
    handlers_slot_start: qd.Tensor = field(int, shape=(CONSTRAINT_HANDLERS + 1,))
    n_vertex_chunks: qd.Tensor = field(int, shape=())
    n_element_chunks: qd.Tensor = field(int, shape=())
    n_env_fem_entries: qd.Tensor = field(int, shape=())
    n_env_link_entries: qd.Tensor = field(int, shape=())
    # Bits where the handler then the environment start in a constraint key (see kernel_order_constraints)
    constraints_key_shift: qd.Tensor = field(int, shape=())
    constraints_env_shift: qd.Tensor = field(int, shape=())
    n_dofs: qd.Tensor = field(int, shape=())
    # The rigid ground plane, -1 without one (see 'SAPCoupler.build')
    ground_geom_idx: qd.Tensor = field(int, shape=())
    substep_dt: qd.Tensor = field(float, shape=())
    dt2: qd.Tensor = field(float, shape=())
    n_newton_iterations: qd.Tensor = field(int, shape=())
    n_pcg_iterations: qd.Tensor = field(int, shape=())
    n_linesearch_iterations: qd.Tensor = field(int, shape=())
    convergence_atol: qd.Tensor = field(float, shape=())
    convergence_rtol: qd.Tensor = field(float, shape=())
    taud: qd.Tensor = field(float, shape=())
    sigma: qd.Tensor = field(float, shape=())
    # The near-rigid factor of the contact regularization, beta^2 / (4 pi^2)
    beta_factor: qd.Tensor = field(float, shape=())
    pcg_threshold: qd.Tensor = field(float, shape=())
    linesearch_ftol: qd.Tensor = field(float, shape=())
    linesearch_max_step_size: qd.Tensor = field(float, shape=())
    linesearch_cost_atol: qd.Tensor = field(float, shape=())
    linesearch_cost_rtol: qd.Tensor = field(float, shape=())
    hydroelastic_stiffness: qd.Tensor = field(float, shape=())
    point_contact_stiffness: qd.Tensor = field(float, shape=())
    rigid_floor_vert_stiffness: qd.Tensor = field(float, shape=())
    cos_angle_threshold: qd.Tensor = field(float, shape=())
    # See CONTACT_GRADIENT_RTOL, CONTACT_AREA_MIN and CAP_RESIDUAL_MAX
    contact_gradient_rtol: qd.Tensor = field(float, shape=())
    contact_area_min: qd.Tensor = field(float, shape=())
    contact_stiffness_min: qd.Tensor = field(float, shape=())
    contact_separation_max: qd.Tensor = field(float, shape=())
    cap_residual_max: qd.Tensor = field(float, shape=())
    # Relative precision of the simulation, below which a ratio of lengths or a norm is round-off
    precision: qd.Tensor = field(float, shape=())


class SAPData(Data):
    """Every array of the SAP coupler, allocated once its scene is built (see 'SAPCoupler.build')."""

    sap_info: SAPInfo
    sap_state: SAPState
    errno: qd.Tensor = field(int, shape=(ENVS,), kind=DataKind.STATE)


class SAPConfig(Config):
    """The compile-time configuration of the kernels of the SAP coupler."""

    has_fem: bool
    has_rigid: bool
    has_equality_constraints: bool
    has_fem_contacts: bool
    has_rigid_contacts: bool
    has_rigid_tet_tree: bool
    enable_fem_self_tet_contact: bool
    enable_rigid_fem_contact: bool
    enable_vertex_constraints: bool
    batch_links_info: bool
    # Whether the lanes of a GPU subgroup take the items of a chunk together (see CHUNK_SIZE), a link summing the wrench
    # of its contacts in several shares (see N_LINK_WRENCH_SHARES), where one CPU thread takes them in a row.
    enable_cooperative_kernels: bool
    # Most dofs the kinematic chain of a contact holds, from CHAIN_DOFS_SIZES
    n_max_chain_dofs: int
    # Key bits the device radix sort orders, and the base-256 logarithm of the key count
    constraints_sort_end_bit: int
    constraints_sort_log256: int


@qd.func
def func_filter_fem_surface_tets(i_t: int, i_a: int, i_q: int, fem_topology: qd.template()) -> bool:
    """Drop a pair of FEM surface tets once per unordered pair, and the pairs sharing a vertex.

    `fem_topology` holds the surface elements and the vertices of the elements of the FEM solver (see FEMInfo).
    """
    surface_elements = qd.static(fem_topology[0])
    el2v = qd.static(fem_topology[1])
    is_dropped = i_a >= i_q
    i_ea = surface_elements[i_a]
    i_eq = surface_elements[i_q]
    i_av = el2v[i_ea]
    i_qv = el2v[i_eq]
    for i, j in qd.static(qd.ndrange(4, 4)):
        if i_av[i] == i_qv[j]:
            is_dropped = True
    return is_dropped


@qd.func
def func_filter_rigid_tets(i_t: int, i_a: int, i_q: int, rigid_tets_pairs: qd.template()) -> bool:
    """Drop a pair of rigid tets whose geoms hold no collision pair.

    `rigid_tets_pairs` holds the geoms of the rigid volume elements and the collision pairs of the geoms (see SAPInfo).
    The tree and the queries hold the same tets, so every pair of geoms is found in both orders, the one of increasing
    geom index alone being kept.
    """
    elems_geom_idx = qd.static(rigid_tets_pairs[0])
    collision_pair_idx = qd.static(rigid_tets_pairs[1])
    i_ag = elems_geom_idx[i_a]
    i_qg = elems_geom_idx[i_q]
    return i_ag >= i_qg or collision_pair_idx[i_ag, i_qg] == -1


class FEMFloorContactType(IntEnum):
    """
    Enum for FEM floor contact types.
    """

    NONE = 0  # No contact
    TET = 1  # Tetrahedral contact
    VERT = 2  # Vertex contact


class RigidFloorContactType(IntEnum):
    """
    Enum for rigid floor contact types.
    """

    NONE = 0  # No contact
    VERT = 1  # Vertex contact
    TET = 2  # Tetrahedral contact


class RigidRigidContactType(IntEnum):
    """
    Enum for rigid-rigid contact types.
    """

    NONE = 0  # No contact
    TET = 1  # Tetrahedral contact


def get_contact_capacities(options: SAPCouplerOptions, n_contacts_max: int, n_candidates_max: int) -> tuple[int, int]:
    """
    Return the contacts and the candidate contacts a contact handler holds in each environment.

    Each is the capacity option of the coupler ('max_contacts' and 'max_collision_pairs'), capped by the most the scene
    can produce, `n_contacts_max` and `n_candidates_max`.
    """
    return min(options.max_contacts, n_contacts_max), min(options.max_collision_pairs, n_candidates_max)


@qd.func
def func_inverse_spd6(A: qd.types.matrix(6, 6)):
    """Return the inverse of a 6x6 symmetric positive-definite matrix, by a Cholesky factorization."""
    L = qd.Matrix.zero(gs.qd_float, 6, 6)
    for j in qd.static(range(6)):
        d = A[j, j]
        for k in qd.static(range(j)):
            d = d - L[j, k] * L[j, k]
        L[j, j] = qd.sqrt(d)
        for i in qd.static(range(j + 1, 6)):
            l = A[i, j]
            for k in qd.static(range(j)):
                l = l - L[i, k] * L[j, k]
            L[i, j] = l / L[j, j]
    # The inverse of L by forward substitution, then L^-T L^-1
    L_inv = qd.Matrix.zero(gs.qd_float, 6, 6)
    for j in qd.static(range(6)):
        for i in qd.static(range(j, 6)):
            l = gs.qd_float(qd.static(float(i == j)))
            for k in qd.static(range(j, i)):
                l = l - L[i, k] * L_inv[k, j]
            L_inv[i, j] = l / L[i, i]
    return L_inv.transpose() @ L_inv


@qd.func
def func_add_slot(
    i_b: int, n_max_per_env: int, envs_n: qd.template(), n: qd.template(), code: qd.template(), errno: qd.Tensor
) -> int:
    """
    Return the slot of a new item of an environment in a list of items of all environments, -1 once it is full.

    The environment is i_b. `envs_n` counts the items of each environment and `n` those of the list. Each environment
    owns `n_max_per_env` slots of the list, so that an environment past its share drops its new items alone, which flags
    `code` in its errors, while the list stays within its capacity.
    """
    i_s = -1
    if qd.atomic_add(envs_n[i_b], 1) < n_max_per_env:
        i_s = qd.atomic_add(n[None], 1)
    else:
        errno[i_b] = errno[i_b] | code
    return i_s


@qd.func
def func_flag_query_overflow(query_results: BVHQueryResults, errno: qd.Tensor):
    """Flag the environments whose tree outgrew its share of the results of a box query (see func_bvh_query_aabb)."""
    n_tree_results_max = query_results.triplets.shape[0] // query_results.trees_count.shape[0]
    for i_b in range(query_results.trees_count.shape[0]):
        if query_results.trees_count[i_b] > n_tree_results_max:
            errno[i_b] = errno[i_b] | SAPErrorCode.OVERFLOW_CANDIDATE_CONTACTS


@qd.func
def tet_barycentric(p, tet_vertices):
    """
    Compute the barycentric coordinates of point p with respect to the tetrahedron defined by tet_vertices.
    tet_vertices is a matrix of shape (3, 4) where each column is a vertex of the tetrahedron.
    """
    v0 = tet_vertices[:, 0]
    v1 = tet_vertices[:, 1]
    v2 = tet_vertices[:, 2]
    v3 = tet_vertices[:, 3]

    # Compute the volumes of the tetrahedra formed by the point and the vertices
    vol_tet_inv = 1.0 / ((v1 - v0).dot((v2 - v0).cross(v3 - v0)))

    # Compute the barycentric coordinates
    b0 = (p - v1).dot((v3 - v1).cross(v2 - v1)) * vol_tet_inv
    b1 = (p - v2).dot((v3 - v2).cross(v0 - v2)) * vol_tet_inv
    b2 = (p - v3).dot((v1 - v3).cross(v0 - v3)) * vol_tet_inv
    b3 = 1.0 - b0 - b1 - b2

    return qd.Vector([b0, b1, b2, b3], dt=gs.qd_float)


def compute_pressure_field(
    verts: np.ndarray, surface_verts: np.ndarray, surface_faces: np.ndarray, modulus: float
) -> np.ndarray:
    """Return the hydroelastic pressure field of a volume mesh at its vertices.

    The field grows linearly with the unsigned distance from each vertex of `verts` to the surface mesh `surface_verts`,
    `surface_faces`, from zero on the surface to `modulus` at the deepest vertex. See
    https://drake.mit.edu/doxygen_cxx/group__hydroelastic__user__guide.html
    """
    distance, *_ = igl.signed_distance(verts, surface_verts, surface_faces, sign_type=igl.SIGNED_DISTANCE_TYPE_UNSIGNED)
    distance = distance.astype(gs.np_float, copy=False)
    distance_max = np.max(distance)
    if distance_max < gs.EPS:
        gs.raise_exception(
            textwrap.dedent(
                f"""\
                Pressure field max distance is too small: {distance_max}. This might be due to a mesh having no internal
                vertices."""
            )
        )
    return distance / distance_max * modulus


class SAPCoupler(ErrnoMixin, CheckpointMixin, RBC):
    """
    This class handles all the coupling between different solvers using the
    Semi-Analytic Primal (SAP) contact solver used in Drake.

    Note
    ----
    For now all batches have the same constraints, such as joint equality constraints are consistent among all batches.
    Paper reference: https://arxiv.org/abs/2110.10107
    Drake reference: https://drake.mit.edu/release_notes/v1.5.0.html
    Code reference: https://github.com/RobotLocomotion/drake/blob/d7a5096c6d0f131705c374390202ad95d0607fd4/multibody/plant/sap_driver.cc
    """

    # Its whole state lives in the arrays 'data' yields (see 'CheckpointMixin')
    is_checkpointable = True
    # Whether the coupler supports a scene with `requires_grad` (see 'Solver.is_differentiable')
    is_differentiable = False

    # ------------------------------------------------------------------------------------
    # --------------------------------- Initialization -----------------------------------
    # ------------------------------------------------------------------------------------

    def __init__(self, simulator: "Simulator", options: "SAPCouplerOptions") -> None:
        self.sim = simulator
        self.options = options
        self.rigid_solver = self.sim.rigid_solver
        self.fem_solver = self.sim.fem_solver
        self._sap_convergence_atol = 0.0 if options.sap_convergence_atol is None else options.sap_convergence_atol
        self._sap_convergence_rtol = options.sap_convergence_rtol
        if self._sap_convergence_rtol is None:
            self._sap_convergence_rtol = CONVERGENCE_RTOL_DEFAULT[gs.np_float]
        if options.fem_floor_contact_type == "tet":
            self._fem_floor_contact_type = FEMFloorContactType.TET
        elif options.fem_floor_contact_type == "vert":
            self._fem_floor_contact_type = FEMFloorContactType.VERT
        elif options.fem_floor_contact_type == "none":
            self._fem_floor_contact_type = FEMFloorContactType.NONE
        else:
            gs.raise_exception(
                textwrap.dedent(
                    f"""\
                    Invalid FEM floor contact type: {options.fem_floor_contact_type}. Must be one of 'tet', 'vert', or
                    'none'."""
                )
            )
        self._enable_fem_self_tet_contact = options.enable_fem_self_tet_contact
        if options.rigid_floor_contact_type == "vert":
            self._rigid_floor_contact_type = RigidFloorContactType.VERT
        elif options.rigid_floor_contact_type == "tet":
            self._rigid_floor_contact_type = RigidFloorContactType.TET
        elif options.rigid_floor_contact_type == "none":
            self._rigid_floor_contact_type = RigidFloorContactType.NONE
        else:
            gs.raise_exception(
                textwrap.dedent(
                    f"""\
                    Invalid rigid floor contact type: {options.rigid_floor_contact_type}. Must be one of 'vert', 'tet'
                    or 'none'."""
                )
            )
        self._enable_rigid_fem_contact = options.enable_rigid_fem_contact

        if options.rigid_rigid_contact_type == "tet":
            self._rigid_rigid_contact_type = RigidRigidContactType.TET
        elif options.rigid_rigid_contact_type == "none":
            self._rigid_rigid_contact_type = RigidRigidContactType.NONE
        else:
            gs.raise_exception(
                f"Invalid rigid-rigid contact type: {options.rigid_rigid_contact_type}. Must be one of 'tet' or 'none'."
            )

        self._rigid_compliant = False
        self._has_fem_pressure = False
        self._has_equality_constraints = False
        self.has_contact = False
        self.sap_data: SAPData | None = None
        self.sap_state: SAPState | None = None
        self.sap_info: SAPInfo | None = None
        self.sap_config: SAPConfig | None = None
        # The contact handlers in the order the kernels run them, the handler of the equality constraints, if any, and
        # every handler of a constraint of the solve, the contact handlers first.
        self.contact_handlers: tuple["BaseContactHandler", ...] = ()
        self.equality_constraint_handler: "RigidConstraintHandler | None" = None
        self.constraint_handlers: tuple["BaseContactHandler | RigidConstraintHandler", ...] = ()

    def build(self) -> None:
        self._B = self.sim._B
        self._enable_rigid_fem_contact &= self.rigid_solver.is_active and self.fem_solver.is_active
        self._enable_fem_self_tet_contact &= self.fem_solver.is_active

        for equality in self.rigid_solver.equalities:
            if equality.type == gs.EQUALITY_TYPE.JOINT and equality.eq_obj2id < 0:
                gs.raise_exception("SAPCoupler does not support JOINT equality constraints without `joint2`.")

        # The floor contacts model a horizontal floor at the current height of the rigid ground plane of each
        # environment, whose orientation is checked at build only, and every other contact leaves that plane out. A
        # scene without one has no floor contact.
        ground_geom_idx = -1
        ground_geoms = [
            geom for geom in self.rigid_solver.geoms if geom.type in (gs.GEOM_TYPE.PLANE, gs.GEOM_TYPE.TERRAIN)
        ]
        if ground_geoms:
            ground_geom, *other_ground_geoms = ground_geoms
            ground_quat = self.rigid_solver.get_geoms_quat(ground_geom.idx)[..., 0, :]
            ground_normal = gu.transform_by_quat(ground_quat.new_tensor(ground_geom.data[:3]), ground_quat)
            if (
                other_ground_geoms
                or ground_geom.type == gs.GEOM_TYPE.TERRAIN
                or not ground_geom.is_fixed
                or not (ground_normal[..., 2] > 1.0 - gs.EPS).all()
            ):
                gs.raise_exception(
                    textwrap.dedent(
                        f"""\
                        SAPCoupler supports a single fixed plane facing up in every environment as ground, got
                        {[(type(geom.entity.morph).__name__, geom.type) for geom in ground_geoms]}."""
                    )
                )
            ground_geom_idx = ground_geom.idx
        else:
            self._fem_floor_contact_type = FEMFloorContactType.NONE
            self._rigid_floor_contact_type = RigidFloorContactType.NONE

        # The rigid triangles touching FEM leave the ground plane out (see 'ground_geom_idx' above)
        rigid_tri_faces_idx = [
            i_f
            for geom in self.rigid_solver.geoms
            if geom.idx != ground_geom_idx
            for i_f in range(geom.face_start, geom.face_end)
        ]
        self._enable_rigid_fem_contact &= len(rigid_tri_faces_idx) > 0

        has_tet_tables = False
        if self.fem_solver.is_active:
            if self.fem_solver._use_implicit_solver is False:
                gs.raise_exception(
                    textwrap.dedent(
                        """\
                        SAPCoupler requires FEM to use implicit solver. Please set `use_implicit_solver=True` in FEM
                        options."""
                    )
                )
            has_tet_tables = (
                self._fem_floor_contact_type == FEMFloorContactType.TET or self._enable_fem_self_tet_contact
            )
            # The rigid-FEM handler reads the FEM pressure field and gradient as the hydroelastic FEM handlers do
            self._has_fem_pressure = has_tet_tables or self._enable_rigid_fem_contact

        rigid_rigid_geoms, floor_geoms, collision_pair_idx = [], [], None
        if self.rigid_solver.is_active:
            # The rigid volumes hold the geoms that a tet handler can put in contact alone: the floor contacts skip the
            # geoms of fixed links, and the rigid-rigid contacts the geoms without a collision pair with another one. A
            # handler left without geoms is dropped, so that no buffer is sized for contacts that cannot occur.
            volume_geoms = [
                geom
                for geom in self.rigid_solver.geoms
                if (geom.contype or geom.conaffinity) and geom.idx != ground_geom_idx
            ]
            collision_pair_idx = qd_to_numpy(self.rigid_solver.collider.collider_info.collision_pair_idx)
            if self._rigid_rigid_contact_type == RigidRigidContactType.TET:
                rigid_rigid_geoms = [
                    geom
                    for geom in volume_geoms
                    if any(
                        [
                            collision_pair_idx[min(geom.idx, other.idx), max(geom.idx, other.idx)] >= 0
                            for other in volume_geoms
                            if other is not geom
                        ]
                    )
                ]
            if self._rigid_floor_contact_type == RigidFloorContactType.TET:
                floor_geoms = [geom for geom in volume_geoms if not geom.link.is_fixed]
            if not volume_geoms and not any([geom.contype or geom.conaffinity for geom in self.rigid_solver.geoms]):
                if (
                    self._rigid_floor_contact_type == RigidFloorContactType.TET
                    or self._rigid_rigid_contact_type == RigidRigidContactType.TET
                ):
                    gs.raise_exception("No rigid collision geometries found.")
            if not rigid_rigid_geoms:
                self._rigid_rigid_contact_type = RigidRigidContactType.NONE
            if self._rigid_floor_contact_type == RigidFloorContactType.TET and not floor_geoms:
                self._rigid_floor_contact_type = RigidFloorContactType.NONE
            if self._rigid_floor_contact_type == RigidFloorContactType.VERT and not volume_geoms:
                self._rigid_floor_contact_type = RigidFloorContactType.NONE
            self._rigid_compliant = (
                self._rigid_floor_contact_type == RigidFloorContactType.TET
                or self._rigid_rigid_contact_type == RigidRigidContactType.TET
            )
            has_tet_tables |= self._rigid_compliant
            # The rigid solver leaves its equality constraints to this coupler, unless constraints are disabled
            # TODO: Dynamically added constraints are not supported for now
            self._has_equality_constraints = (
                self.rigid_solver.n_equalities > 0 and not self.rigid_solver._options.disable_constraint
            )
        has_rigid_rigid_contact = (
            self.rigid_solver.is_active and self._rigid_rigid_contact_type == RigidRigidContactType.TET
        )

        fem_pressure = []
        if self._has_fem_pressure:
            for fem_entity in self.fem_solver.entities:
                init_positions = tensor_to_array(fem_entity.init_positions)
                fem_pressure.append(
                    compute_pressure_field(
                        init_positions,
                        init_positions,
                        fem_entity.surface_triangles,
                        fem_entity.material.hydroelastic_modulus,
                    )
                )

        # The geoms of the rigid-rigid contacts come first, the tree of their tets spanning the first elements
        rigid_volume = dict(verts=[], elems=[], verts_geom_idx=[], elems_geom_idx=[], pressure_field=[])
        n_rigid_volume_verts = n_rigid_volume_elems = n_rigid_rigid_volume_elems = n_floor_volume_elems = 0
        if self._rigid_compliant:
            floor_only_geoms = [geom for geom in floor_geoms if geom not in rigid_rigid_geoms]
            rigid_rigid_geoms_idx = {geom.idx for geom in rigid_rigid_geoms}
            for geom in (*rigid_rigid_geoms, *floor_only_geoms):
                volume = geom.trimesh.volume
                tet_cfg = {"nobisect": False, "maxvolume": volume / 100}
                mesh_verts, mesh_elems = eu.mesh_to_elements(geom.trimesh, tet_cfg=tet_cfg)
                verts, elems = eu.split_all_surface_tets(mesh_verts, mesh_elems)
                rigid_volume["verts"].append(verts)
                rigid_volume["elems"].append(elems + n_rigid_volume_verts)
                rigid_volume["verts_geom_idx"].append(np.full(len(verts), geom.idx, dtype=gs.np_int))
                rigid_volume["elems_geom_idx"].append(np.full(len(elems), geom.idx, dtype=gs.np_int))
                rigid_volume["pressure_field"].append(
                    compute_pressure_field(verts, geom.init_verts, geom.init_faces, self.options.hydroelastic_stiffness)
                )
                if geom.idx in rigid_rigid_geoms_idx:
                    n_rigid_rigid_volume_elems += len(elems)
                if not geom.link.is_fixed:
                    n_floor_volume_elems += len(elems)
                n_rigid_volume_verts += len(verts)
                n_rigid_volume_elems += len(elems)
        rigid_volume_elems_geom_idx = np.concatenate([np.zeros(0, dtype=gs.np_int), *rigid_volume["elems_geom_idx"]])

        # The contact handlers accumulate into the same gradients, so their order sets the rounding of the solve
        options = self.options
        contact_handlers = []
        if self.fem_solver.is_active:
            if self._fem_floor_contact_type == FEMFloorContactType.TET:
                contact_handlers.append(FEMFloorTetContactHandler(self.sim, options))
            if self._fem_floor_contact_type == FEMFloorContactType.VERT:
                contact_handlers.append(FEMFloorVertContactHandler(self.sim, options))
            if self._enable_fem_self_tet_contact:
                contact_handlers.append(FEMSelfTetContactHandler(self.sim, options))
        if self.rigid_solver.is_active:
            if self._rigid_floor_contact_type == RigidFloorContactType.VERT:
                contact_handlers.append(RigidFloorVertContactHandler(self.sim, options))
            elif self._rigid_floor_contact_type == RigidFloorContactType.TET:
                contact_handlers.append(
                    RigidFloorTetContactHandler(self.sim, options, n_rigid_volume_elems, n_floor_volume_elems)
                )
            if has_rigid_rigid_contact:
                contact_handlers.append(
                    RigidRigidTetContactHandler(
                        self.sim, options, n_rigid_volume_elems, rigid_volume_elems_geom_idx, collision_pair_idx
                    )
                )
        if self._enable_rigid_fem_contact:
            contact_handlers.append(RigidFemTriTetContactHandler(self.sim, options))
        self.contact_handlers = tuple(contact_handlers)
        if self._has_equality_constraints:
            self.equality_constraint_handler = RigidConstraintHandler(self.sim)
        self.constraint_handlers = (
            *self.contact_handlers,
            *([self.equality_constraint_handler] if self._has_equality_constraints else []),
        )

        # The chunks of the FEM vertices of each entity, then of the rigid dofs and the FEM elements (see CHUNK_SIZE)
        chunks_vertex_start, chunks_vertex_end, entities_chunk_start = [], [], [0]
        entities_n_vertices = [entity.n_vertices for entity in self.fem_solver.entities]
        if self.fem_solver.is_active:
            entities_vertex_start = np.cumsum([0, *entities_n_vertices])
            for i_v_start, i_v_end in zip(entities_vertex_start[:-1], entities_vertex_start[1:]):
                for i_v in range(i_v_start, i_v_end, CHUNK_SIZE):
                    chunks_vertex_start.append(i_v)
                    chunks_vertex_end.append(min(i_v + CHUNK_SIZE, i_v_end))
                entities_chunk_start.append(len(chunks_vertex_start))
        n_vertex_chunks = len(chunks_vertex_start)
        n_dof_chunks = math.ceil(self.rigid_solver.n_dofs / CHUNK_SIZE) if self.rigid_solver.is_active else 0
        n_element_chunks = math.ceil(self.fem_solver.n_elements / CHUNK_SIZE) if self.fem_solver.is_active else 0

        # The slots of the constraints of each handler start at the total capacity of the handlers before it. The key of
        # a constraint is its environment, then its handler, then its key in the handler. The radix sort orders a
        # positive even multiple of 8 bits.
        handlers_slot_start = np.cumsum([0, *[handler.max_constraints for handler in self.constraint_handlers]])
        n_slots = int(handlers_slot_start[-1])
        key_bits = max([handler.key_bits for handler in self.constraint_handlers] + [1])
        handler_bits = max(1, len(self.constraint_handlers) - 1).bit_length()
        env_bits = max(1, self._B - 1).bit_length()
        assert env_bits + handler_bits + key_bits <= 64
        # Outside performance mode, the device radix sort orders the widest keys, with a scan depth reaching any count
        # of them a device holds, so that its kernels serve every scene.
        constraints_sort_end_bit = 16 * math.ceil((env_bits + handler_bits + key_bits) / 16)
        constraints_sort_log256 = max(1, math.ceil(math.log(max(n_slots, 2), 256)))
        if gs.use_ndarray:
            constraints_sort_end_bit = 64
            constraints_sort_log256 = max(constraints_sort_log256, 4)
        is_device_sort = gs.backend != gs.cpu
        # The chunks of the sorted constraints of each environment
        n_units = self._B * math.ceil(n_slots // self._B / CHUNK_SIZE) if n_slots >= self._B else 1
        # The contacts of each FEM vertex and rigid link, each environment owning the capacity of its contacts
        n_env_fem_entries = sum(
            [handler.max_constraints // self._B * handler.n_fem_corners for handler in self.constraint_handlers]
        )
        n_env_link_entries = sum(
            [handler.max_constraints // self._B * handler.n_rigid_sides for handler in self.constraint_handlers]
        )
        has_rigid_contacts = n_env_link_entries > 0
        # The most dofs a link and its ancestors hold, which bounds the kinematic chain of a contact that
        # func_link_point_delassus keeps in registers, so that the bound is static. It snaps to a bounded set, and
        # outside performance mode to its largest value, whose kernels serve every scene.
        n_chain_dofs_max = 1
        for link in self.rigid_solver.links if self.rigid_solver.is_active else ():
            n_chain_dofs, i_l = link.n_dofs, link.parent_idx
            while i_l > -1:
                n_chain_dofs += self.rigid_solver.links[i_l].n_dofs
                i_l = self.rigid_solver.links[i_l].parent_idx
            n_chain_dofs_max = max(n_chain_dofs_max, n_chain_dofs)
        if n_chain_dofs_max > CHAIN_DOFS_SIZES[-1]:
            gs.raise_exception(
                f"SAPCoupler supports kinematic chains of {CHAIN_DOFS_SIZES[-1]} dofs at most, a link of the scene and "
                f"its ancestors holding {n_chain_dofs_max}."
            )
        n_max_chain_dofs = CHAIN_DOFS_SIZES[-1]
        if not gs.use_ndarray:
            n_max_chain_dofs = min([size for size in CHAIN_DOFS_SIZES if size >= n_chain_dofs_max])
        enable_cooperative_kernels = gs.backend != gs.cpu
        n_link_wrench_shares = N_LINK_WRENCH_SHARES if enable_cooperative_kernels else 1

        # The subtree of each link spans the links up to its end, every link following its parent in a depth-first
        # order, and each dof sums the wrenches of the subtree of its link (see func_gather_rigid_dof).
        n_links = self.rigid_solver.n_links if self.rigid_solver.is_active else 0
        n_dofs = self.rigid_solver.n_dofs if self.rigid_solver.is_active else 0
        links_subtree_end = np.arange(1, n_links + 1, dtype=gs.np_int)
        dofs_link_idx = np.zeros(n_dofs, dtype=gs.np_int)
        dofs_root_link_idx = np.zeros(n_dofs, dtype=gs.np_int)
        if self.rigid_solver.is_active:
            for link in reversed(self.rigid_solver.links):
                if link.parent_idx >= 0:
                    links_subtree_end[link.parent_idx] = max(
                        links_subtree_end[link.parent_idx], links_subtree_end[link.idx]
                    )
            for link in self.rigid_solver.links:
                dofs_link_idx[link.dof_start : link.dof_end] = link.idx
                i_root = link.idx
                while self.rigid_solver.links[i_root].parent_idx >= 0:
                    i_root = self.rigid_solver.links[i_root].parent_idx
                dofs_root_link_idx[link.dof_start : link.dof_end] = i_root

        sizes = dict(
            envs=self._B,
            vertices=self.fem_solver.n_vertices,
            elements=self.fem_solver.n_elements,
            dofs=self.rigid_solver.n_dofs,
            links=self.rigid_solver.n_links,
            geoms=self.rigid_solver.n_geoms,
            rigid_volume_verts=n_rigid_volume_verts,
            rigid_volume_elems=n_rigid_volume_elems,
            rigid_tri_faces=len(rigid_tri_faces_idx),
            fem_entities=len(entities_n_vertices),
            vertex_chunks=n_vertex_chunks,
            vertex_dof_chunks=n_vertex_chunks + n_dof_chunks,
            element_chunks=n_element_chunks,
            constraint_handlers=len(self.constraint_handlers),
            constraint_slots=n_slots,
            constraint_units=n_units,
            constraint_sort_slots=sort_scratch_slots(n_slots, constraints_sort_log256) if is_device_sort else 0,
            fem_incidence=self._B * n_env_fem_entries,
            link_incidence=self._B * n_env_link_entries,
            link_wrench_shares=n_links * n_link_wrench_shares,
        )
        precision = np.finfo(gs.np_float).eps
        self.sap_data = SAPData(
            sizes=sizes,
            values=dict(
                sap_info=dict(
                    n_vertex_chunks=n_vertex_chunks,
                    n_element_chunks=n_element_chunks,
                    n_env_fem_entries=n_env_fem_entries,
                    n_env_link_entries=n_env_link_entries,
                    constraints_key_shift=key_bits,
                    constraints_env_shift=key_bits + handler_bits,
                    n_dofs=self.rigid_solver.n_dofs,
                    ground_geom_idx=ground_geom_idx,
                    substep_dt=self.sim._substep_dt,
                    dt2=self.sim._substep_dt**2,
                    n_newton_iterations=options.n_sap_iterations,
                    n_pcg_iterations=options.n_pcg_iterations,
                    n_linesearch_iterations=options.n_linesearch_iterations,
                    convergence_atol=self._sap_convergence_atol,
                    convergence_rtol=self._sap_convergence_rtol,
                    taud=options.sap_taud,
                    sigma=options.sap_sigma,
                    beta_factor=options.sap_beta**2 / (4.0 * math.pi**2),
                    # A residual of the conjugate gradient below the precision of the simulation is round-off, iterating
                    # on which makes its products underflow.
                    pcg_threshold=max(options.pcg_threshold, precision**2),
                    linesearch_ftol=options.linesearch_ftol,
                    linesearch_max_step_size=options.linesearch_max_step_size,
                    linesearch_cost_atol=LINESEARCH_COST_ATOL,
                    linesearch_cost_rtol=LINESEARCH_COST_RTOL * precision / np.finfo(np.float64).eps,
                    hydroelastic_stiffness=options.hydroelastic_stiffness,
                    point_contact_stiffness=options.point_contact_stiffness,
                    rigid_floor_vert_stiffness=1.0e6,
                    cos_angle_threshold=COS_ANGLE_THRESHOLD,
                    contact_gradient_rtol=CONTACT_GRADIENT_RTOL * precision,
                    contact_area_min=CONTACT_AREA_MIN,
                    contact_stiffness_min=CONTACT_STIFFNESS_MIN,
                    contact_separation_max=CONTACT_SEPARATION_MAX,
                    cap_residual_max=CAP_RESIDUAL_MAX,
                    precision=precision,
                )
            ),
            env_first={
                (
                    SAPData.sap_state.vertices_dofs_partials,
                    SAPData.sap_state.elements_partials,
                    SAPData.sap_state.fem_corner_forces,
                    SAPData.sap_state.constraint_order.fem_incidence_start,
                    SAPData.sap_state.constraint_order.link_incidence_start,
                ): True,
            },
            unused=dict(
                fem=not self.fem_solver.is_active,
                rigid=not self.rigid_solver.is_active,
                fem_pressure=not self._has_fem_pressure,
                rigid_volume=not self._rigid_compliant,
                rigid_rigid=not has_rigid_rigid_contact,
                rigid_fem=not self._enable_rigid_fem_contact,
                tet_tables=not has_tet_tables,
                rigid_contacts=not has_rigid_contacts,
                host_sort=is_device_sort,
                device_sort=not is_device_sort,
            ),
        )
        self.sap_state, self.sap_info = self.sap_data.sap_state, self.sap_data.sap_info
        sap_info = self.sap_info
        if self._has_fem_pressure:
            sap_info.fem_pressure.from_numpy(np.concatenate(fem_pressure, dtype=gs.np_float))
        if self._rigid_compliant:
            sap_info.rigid_volume_verts_rest.from_numpy(np.concatenate(rigid_volume["verts"], dtype=gs.np_float))
            sap_info.rigid_volume_verts_geom_idx.from_numpy(np.concatenate(rigid_volume["verts_geom_idx"]))
            sap_info.rigid_pressure_field.from_numpy(np.concatenate(rigid_volume["pressure_field"], dtype=gs.np_float))
            sap_info.rigid_volume_elems.from_numpy(np.concatenate(rigid_volume["elems"], dtype=gs.np_int))
            sap_info.rigid_volume_elems_geom_idx.from_numpy(rigid_volume_elems_geom_idx)
            kernel_compute_rigid_pressure_gradient_rest(sap_info)
        if has_rigid_rigid_contact:
            sap_info.rigid_collision_pair_idx.from_numpy(collision_pair_idx)
        if self._enable_rigid_fem_contact:
            sap_info.rigid_tri_faces_idx.from_numpy(np.array(rigid_tri_faces_idx, dtype=gs.np_int))
        if has_tet_tables:
            sap_info.marching_tets_edge_table.from_numpy(np.array(MARCHING_TETS_EDGE_TABLE, dtype=gs.np_int))
            sap_info.tet_edges.from_numpy(np.array(TET_EDGES, dtype=gs.np_int))
        if self.fem_solver.is_active:
            sap_info.chunks_vertex_start.from_numpy(np.array(chunks_vertex_start, dtype=gs.np_int))
            sap_info.chunks_vertex_end.from_numpy(np.array(chunks_vertex_end, dtype=gs.np_int))
            sap_info.entities_chunk_start.from_numpy(np.array(entities_chunk_start, dtype=gs.np_int))
            sap_info.fem_vertices_entity_idx.from_numpy(
                np.repeat(np.arange(len(entities_n_vertices), dtype=gs.np_int), entities_n_vertices)
            )
            sap_info.fem_corners_rank.from_numpy(
                np.argsort(qd_to_numpy(self.fem_solver.fem_info.vertices_corners_idx), kind="stable").astype(
                    gs.np_int, copy=False
                )
            )
        if has_rigid_contacts:
            sap_info.links_subtree_end.from_numpy(links_subtree_end)
            sap_info.dofs_link_idx.from_numpy(dofs_link_idx)
            sap_info.dofs_root_link_idx.from_numpy(dofs_root_link_idx)
        sap_info.handlers_slot_start.from_numpy(handlers_slot_start.astype(gs.np_int))
        self.sap_config = SAPConfig(
            has_fem=self.fem_solver.is_active,
            has_rigid=self.rigid_solver.is_active,
            has_equality_constraints=self._has_equality_constraints,
            has_fem_contacts=n_env_fem_entries > 0,
            has_rigid_contacts=has_rigid_contacts,
            has_rigid_tet_tree=has_rigid_rigid_contact,
            enable_fem_self_tet_contact=self._enable_fem_self_tet_contact,
            enable_rigid_fem_contact=self._enable_rigid_fem_contact,
            enable_vertex_constraints=self.fem_solver.fem_config.enable_vertex_constraints,
            batch_links_info=self.rigid_solver._options.batch_links_info,
            enable_cooperative_kernels=enable_cooperative_kernels,
            n_max_chain_dofs=n_max_chain_dofs,
            constraints_sort_end_bit=constraints_sort_end_bit,
            constraints_sort_log256=constraints_sort_log256,
        )
        if self._has_equality_constraints:
            kernel_build_equality_constraints(
                self.rigid_solver.dyn_info.equalities,
                self.rigid_solver.dyn_info.joints,
                sap_info,
                self.rigid_solver.rigid_config,
                self.equality_constraint_handler,
            )

        # The errors of each env, raised at the beginning of the next step (see 'ErrnoMixin.register_errors')
        self.build_errno(self._B)
        envs_max_contacts = ", ".join(
            [f"{contact.name}: {contact.max_contact_pairs // self._B}" for contact in self.contact_handlers]
        )
        envs_max_candidates = ", ".join(
            [
                f"{contact.name}: {contact.max_contact_candidates // self._B}"
                for contact in self.contact_handlers
                if contact.has_candidates
            ]
        )
        self.register_errors(
            self.sap_data.errno,
            {
                SAPErrorCode.OVERFLOW_CANDIDATE_CONTACTS: (
                    "Exceeding max number of SAP collision pairs of a kind of contact in an environment "
                    f"({envs_max_candidates}). Please increase the value of SAPCoupler's option 'max_collision_pairs'."
                ),
                SAPErrorCode.OVERFLOW_CONTACTS: (
                    "Exceeding max number of SAP contacts of a kind of contact in an environment "
                    f"({envs_max_contacts}). Please increase the value of SAPCoupler's option 'max_contacts'."
                ),
                SAPErrorCode.INVALID_SOLVE: (
                    "SAP solve stopped at its Newton iteration cap with a residual |gradient| / max(|momentum|, "
                    f"|impulse|) of {CAP_RESIDUAL_MAX} or more, which leaves its velocities meaningless. Please "
                    "increase the value of SAPCoupler's option 'n_sap_iterations'."
                ),
            },
        )

        self._init_bvh(n_rigid_rigid_volume_elems)

    def reset(self, envs_idx=None):
        # The state of the coupler, its errors included, is written back by 'set_state', which precedes the restart
        pass

    @property
    def data(self) -> Iterator[DataItem]:
        """Yield every array the coupler holds, under the dotted name a checkpoint uses for it (see 'Solver.data')."""
        # A coupler whose build failed holds no array
        if self.sap_data is None:
            return
        yield from iter_data(self.sap_data, "errno", DataKind.STATE)

    def _init_bvh(self, n_rigid_tets: int) -> None:
        """Allocate the tree set and the box query of each contact handler traversing a tree."""
        # The contact kernel takes the queries as one struct, so a disabled handler keeps its members, switched off. The
        # surface tet boxes serve the self-contact tree and the rigid triangle query alike.
        handlers = {type(contact): contact for contact in self.contact_handlers}
        has_fem_tet_tree = self._enable_fem_self_tet_contact or self._enable_rigid_fem_contact
        n_tets = self.fem_solver.n_surface_elements if has_fem_tet_tree else 0
        self.fem_surface_tet_bvh_state, self.fem_surface_tet_bvh_config = get_bvh_data(
            self.sim._B, n_tets, is_active=has_fem_tet_tree
        )
        # The results of the query of a handler take the capacity of its candidates, each environment holding an equal
        # share of them (see func_bvh_query_aabb).
        fem_self = handlers.get(FEMSelfTetContactHandler)
        # The surface tets of each environment queried against their own tree
        fem_self_query = BVHQueryState(
            leaves=self.fem_surface_tet_bvh_state.leaves,
            tree=self.fem_surface_tet_bvh_state.tree,
            results=BVHQueryResults(
                sizes=dict(
                    query_results=0 if fem_self is None else fem_self.max_contact_candidates, bvh_trees=self.sim._B
                ),
                unused=fem_self is None,
            ),
        )

        n_rigid_tris = self.sap_info.rigid_tri_faces_idx.shape[0] if self._enable_rigid_fem_contact else 0
        self.rigid_tri_bvh_state, self.rigid_tri_bvh_config = get_bvh_data(
            self.sim._B, n_rigid_tris, is_active=self._enable_rigid_fem_contact
        )
        rigid_fem = handlers.get(RigidFemTriTetContactHandler)
        # The surface tets of each environment queried against the rigid triangle tree of that environment
        rigid_tri_query = BVHQueryState(
            leaves=self.fem_surface_tet_bvh_state.leaves,
            tree=self.rigid_tri_bvh_state.tree,
            results=BVHQueryResults(
                sizes=dict(
                    query_results=0 if rigid_fem is None else rigid_fem.max_contact_candidates, bvh_trees=self.sim._B
                ),
                unused=rigid_fem is None,
            ),
        )

        rigid_rigid = handlers.get(RigidRigidTetContactHandler)
        self.rigid_tet_bvh_state, self.rigid_tet_bvh_config = get_bvh_data(
            self.sim._B, n_rigid_tets, is_active=rigid_rigid is not None
        )
        # The rigid tets of each environment queried against their own tree
        rigid_tet_query = BVHQueryState(
            leaves=self.rigid_tet_bvh_state.leaves,
            tree=self.rigid_tet_bvh_state.tree,
            results=BVHQueryResults(
                sizes=dict(
                    query_results=0 if rigid_rigid is None else rigid_rigid.max_contact_candidates,
                    bvh_trees=self.sim._B,
                ),
                unused=rigid_rigid is None,
            ),
        )
        self.contact_queries_state = SAPContactQueriesState(
            fem_self=fem_self_query, rigid_tri=rigid_tri_query, rigid_tet=rigid_tet_query
        )

    # ------------------------------------------------------------------------------------
    # -------------------------------------- Main ----------------------------------------
    # ------------------------------------------------------------------------------------

    def preprocess(self):
        rigid_solver, fem_solver = self.rigid_solver, self.fem_solver
        if self._has_fem_pressure:
            kernel_compute_fem_pressure_gradient(
                fem_solver.fem_state, self.sap_state, fem_solver.fem_info, self.sap_info
            )
        if rigid_solver.is_active:
            kernel_update_all_verts(rigid_solver.dyn_state, rigid_solver.dyn_info, rigid_solver.rigid_config)
        if self._rigid_compliant:
            kernel_update_rigid_volume_verts_pressure_gradient(
                rigid_solver.dyn_state.geoms, self.sap_state, self.sap_info
            )

        if self._enable_fem_self_tet_contact or self._enable_rigid_fem_contact:
            kernel_compute_fem_surface_tet_aabb(
                fem_solver.fem_state, self.fem_surface_tet_bvh_state.leaves, fem_solver.fem_info
            )
        if self._enable_fem_self_tet_contact:
            build_bvh(self.fem_surface_tet_bvh_state, self.fem_surface_tet_bvh_config, eps=gs.EPS)
        if self._enable_rigid_fem_contact:
            kernel_compute_rigid_tri_aabb(
                rigid_solver.dyn_state.free_verts,
                rigid_solver.dyn_state.fixed_verts,
                self.rigid_tri_bvh_state.leaves,
                rigid_solver.dyn_info.faces,
                rigid_solver.dyn_info.verts,
                self.sap_info,
            )
            build_bvh(self.rigid_tri_bvh_state, self.rigid_tri_bvh_config, eps=gs.EPS)
        if self.sap_config.has_rigid_tet_tree:
            kernel_compute_rigid_tet_aabb(self.sap_state, self.rigid_tet_bvh_state.leaves, self.sap_info)
            build_bvh(self.rigid_tet_bvh_state, self.rigid_tet_bvh_config, eps=gs.EPS)

        self.has_contact = kernel_update_contact(
            fem_solver.fem_state,
            rigid_solver.dyn_state.geoms,
            rigid_solver.dyn_state.free_verts,
            rigid_solver.dyn_state.fixed_verts,
            self.sap_state,
            self.contact_queries_state,
            fem_solver.fem_info,
            rigid_solver.dyn_info.links,
            rigid_solver.dyn_info.faces,
            rigid_solver.dyn_info.verts,
            rigid_solver.dyn_info.geoms,
            self.sap_info,
            rigid_solver.rigid_config,
            self.sap_config,
            self.contact_handlers,
            self.constraint_handlers,
            self.sap_data.errno,
        )
        kernel_order_constraints(
            self.sap_state, fem_solver.fem_info, self.sap_info, self.sap_config, self.constraint_handlers
        )

    def couple(self):
        # Equality constraints bind the rigid dofs whether anything is in contact or not
        if self.has_contact or self._has_equality_constraints:
            fem_solver, rigid_solver = self.fem_solver, self.rigid_solver
            fem_state, fem_info = fem_solver.fem_state, fem_solver.fem_info
            vertex_constraints, fem_config = fem_solver.vertex_constraints, fem_solver.fem_config
            dofs_state, links_state = rigid_solver.dyn_state.dofs, rigid_solver.dyn_state.links
            entities_info, links_info = rigid_solver.dyn_info.entities, rigid_solver.dyn_info.links
            rigid_info, rigid_config = rigid_solver.rigid_info, rigid_solver.rigid_config
            loops = self.sap_state.loops
            kernel_init_sap_solve(
                loops.newton_counter,
                fem_state,
                dofs_state,
                self.sap_state,
                fem_info,
                vertex_constraints,
                self.sap_info,
                fem_config,
                self.sap_config,
            )
            # The regularization reads the preconditioner of the FEM solve and the factor of the rigid mass matrix of
            # this substep, which the solvers compute after the detection of the contacts, and the velocities the solve
            # starts from
            kernel_compute_regularization(
                fem_state,
                dofs_state,
                links_state,
                self.sap_state,
                fem_info,
                entities_info,
                links_info,
                rigid_solver.dyn_info.equalities,
                rigid_info,
                self.sap_info,
                rigid_config,
                self.sap_config,
                self.contact_handlers,
                self.equality_constraint_handler,
            )
            kernel_sap_solve(
                loops.newton_counter,
                loops.pcg_counter,
                loops.linesearch_counter,
                fem_state,
                dofs_state,
                links_state,
                self.sap_state,
                fem_info,
                vertex_constraints,
                links_info,
                rigid_info,
                self.sap_info,
                fem_config,
                rigid_config,
                self.sap_config,
                self.constraint_handlers,
                self.equality_constraint_handler,
            )
            kernel_update_vel(
                fem_state,
                dofs_state,
                self.sap_state,
                self.sap_info,
                self.sap_config,
                self.contact_handlers,
                self.sap_data.errno,
            )

    # ------------------------------------------------------------------------------------
    # ----------------------------------- Properties -------------------------------------
    # ------------------------------------------------------------------------------------
    @property
    def active_solvers(self):
        """All the active solvers managed by the scene's simulator."""
        return self.sim.active_solvers


# ------------------------------------------------------------------------------------
# ---------------------------------- Pressure fields ---------------------------------
# ------------------------------------------------------------------------------------


@qd.kernel
def kernel_compute_rigid_pressure_gradient_rest(sap_info: SAPInfo):
    """Compute the gradient of the pressure field of every rigid volume element in the frame of its geom."""
    for i_e in range(sap_info.rigid_volume_elems.shape[0]):
        sap_info.rigid_pressure_gradient_rest[i_e] = qd.Vector.zero(gs.qd_float, 3)
        for i in qd.static(range(4)):
            i_v0 = sap_info.rigid_volume_elems[i_e][i]
            i_v1 = sap_info.rigid_volume_elems[i_e][(i + 1) % 4]
            i_v2 = sap_info.rigid_volume_elems[i_e][(i + 2) % 4]
            i_v3 = sap_info.rigid_volume_elems[i_e][(i + 3) % 4]
            pos_v0 = sap_info.rigid_volume_verts_rest[i_v0]
            pos_v1 = sap_info.rigid_volume_verts_rest[i_v1]
            pos_v2 = sap_info.rigid_volume_verts_rest[i_v2]
            pos_v3 = sap_info.rigid_volume_verts_rest[i_v3]

            e10 = pos_v0 - pos_v1
            e12 = pos_v2 - pos_v1
            e13 = pos_v3 - pos_v1

            area_vector = e12.cross(e13)
            signed_volume = area_vector.dot(e10)
            if qd.abs(signed_volume) > sap_info.precision[None] * e10.norm() * e12.norm() * e13.norm():
                grad_i = area_vector / signed_volume
                sap_info.rigid_pressure_gradient_rest[i_e] = (
                    sap_info.rigid_pressure_gradient_rest[i_e] + grad_i * sap_info.rigid_pressure_field[i_v0]
                )


@qd.kernel
def kernel_update_rigid_volume_verts_pressure_gradient(geoms_state: GeomsState, sap_state: SAPState, sap_info: SAPInfo):
    """Pose the vertices and the pressure gradients of the rigid volume meshes with their geoms."""
    for i_v, i_b in qd.ndrange(
        sap_info.rigid_volume_verts_rest.shape[0],
        sap_state.rigid_volume_verts.shape[1],
        axes=qd.static(eoa(sap_state.rigid_volume_verts)),
    ):
        i_g = sap_info.rigid_volume_verts_geom_idx[i_v]
        pos = geoms_state.pos[i_g, i_b]
        quat = geoms_state.quat[i_g, i_b]
        R = gu.qd_quat_to_R(quat, sap_info.precision[None])
        sap_state.rigid_volume_verts[i_v, i_b] = R @ sap_info.rigid_volume_verts_rest[i_v] + pos

    for i_e, i_b in qd.ndrange(
        sap_info.rigid_volume_elems.shape[0],
        sap_state.rigid_pressure_gradient.shape[1],
        axes=qd.static(eoa(sap_state.rigid_pressure_gradient)),
    ):
        i_g = sap_info.rigid_volume_elems_geom_idx[i_e]
        quat = geoms_state.quat[i_g, i_b]
        R = gu.qd_quat_to_R(quat, sap_info.precision[None])
        sap_state.rigid_pressure_gradient[i_e, i_b] = R @ sap_info.rigid_pressure_gradient_rest[i_e]


@qd.kernel
def kernel_compute_fem_pressure_gradient(
    fem_state: FEMState, sap_state: SAPState, fem_info: FEMInfo, sap_info: SAPInfo
):
    """Compute the gradient of the pressure field of every FEM element at its current shape."""
    for i_e, i_b in qd.ndrange(
        fem_info.elements.V.shape[0],
        sap_state.fem_pressure_gradient.shape[1],
        axes=qd.static(eoa(sap_state.fem_pressure_gradient)),
    ):
        sap_state.fem_pressure_gradient[i_e, i_b] = qd.Vector.zero(gs.qd_float, 3)

        for i in qd.static(range(4)):
            i_v0 = fem_info.elements.el2v[i_e][i]
            i_v1 = fem_info.elements.el2v[i_e][(i + 1) % 4]
            i_v2 = fem_info.elements.el2v[i_e][(i + 2) % 4]
            i_v3 = fem_info.elements.el2v[i_e][(i + 3) % 4]
            pos_v0 = fem_state.vertices.pos[i_v0, i_b]
            pos_v1 = fem_state.vertices.pos[i_v1, i_b]
            pos_v2 = fem_state.vertices.pos[i_v2, i_b]
            pos_v3 = fem_state.vertices.pos[i_v3, i_b]

            e10 = pos_v0 - pos_v1
            e12 = pos_v2 - pos_v1
            e13 = pos_v3 - pos_v1

            area_vector = e12.cross(e13)
            signed_volume = area_vector.dot(e10)
            if qd.abs(signed_volume) > sap_info.precision[None] * e10.norm() * e12.norm() * e13.norm():
                grad_i = area_vector / signed_volume
                sap_state.fem_pressure_gradient[i_e, i_b] = (
                    sap_state.fem_pressure_gradient[i_e, i_b] + grad_i * sap_info.fem_pressure[i_v0]
                )


# ------------------------------------------------------------------------------------
# -------------------------------------- BVH -----------------------------------------
# ------------------------------------------------------------------------------------


@qd.kernel
def kernel_compute_fem_surface_tet_aabb(fem_state: FEMState, bvh_leaves: BVHLeaves, fem_info: FEMInfo):
    """Fit the box of every FEM surface tet, the leaves of the trees of the FEM contacts."""
    for i_se, i_b in qd.ndrange(
        fem_info.surface_elements.shape[0], bvh_leaves.aabbs_min.shape[0], axes=qd.static(eoa(fem_state.vertices.pos))
    ):
        i_e = fem_info.surface_elements[i_se]
        i_vs = fem_info.elements.el2v[i_e]

        leaf_min = qd.Vector([np.inf, np.inf, np.inf], dt=gs.qd_float)
        leaf_max = -leaf_min
        for i in qd.static(range(4)):
            pos_v = fem_state.vertices.pos[i_vs[i], i_b]
            leaf_min = qd.min(leaf_min, pos_v)
            leaf_max = qd.max(leaf_max, pos_v)
        bvh_leaves.aabbs_min[i_b, i_se] = leaf_min
        bvh_leaves.aabbs_max[i_b, i_se] = leaf_max


@qd.kernel
def kernel_compute_rigid_tri_aabb(
    free_verts_state: VertsState,
    fixed_verts_state: VertsState,
    bvh_leaves: BVHLeaves,
    faces_info: FacesInfo,
    verts_info: VertsInfo,
    sap_info: SAPInfo,
):
    """Fit the box of every rigid triangle touching FEM, the leaves of the trees of the rigid-FEM contacts."""
    for i_t, i_b in qd.ndrange(
        sap_info.rigid_tri_faces_idx.shape[0], bvh_leaves.aabbs_min.shape[0], axes=qd.static(eoa(free_verts_state.pos))
    ):
        i_f = sap_info.rigid_tri_faces_idx[i_t]
        tri_vertices = qd.Matrix.zero(gs.qd_float, 3, 3)
        for i in qd.static(range(3)):
            i_v = faces_info.verts_idx[i_f][i]
            i_fv = verts_info.verts_state_idx[i_v]
            if verts_info.is_fixed[i_v]:
                tri_vertices[:, i] = fixed_verts_state.pos[i_fv]
            else:
                tri_vertices[:, i] = free_verts_state.pos[i_fv, i_b]
        pos_v0, pos_v1, pos_v2 = tri_vertices[:, 0], tri_vertices[:, 1], tri_vertices[:, 2]

        bvh_leaves.aabbs_min[i_b, i_t] = qd.min(pos_v0, pos_v1, pos_v2)
        bvh_leaves.aabbs_max[i_b, i_t] = qd.max(pos_v0, pos_v1, pos_v2)


@qd.kernel
def kernel_compute_rigid_tet_aabb(sap_state: SAPState, bvh_leaves: BVHLeaves, sap_info: SAPInfo):
    """Fit the box of every rigid tet of the rigid-rigid contacts, the leaves of their trees.

    The geoms of the rigid-rigid contacts come first in the rigid volume, their tets spanning the leaves of the trees.
    """
    for i_e, i_b in qd.ndrange(
        bvh_leaves.aabbs_min.shape[1], bvh_leaves.aabbs_min.shape[0], axes=qd.static(eoa(sap_state.rigid_volume_verts))
    ):
        i_v0 = sap_info.rigid_volume_elems[i_e][0]
        i_v1 = sap_info.rigid_volume_elems[i_e][1]
        i_v2 = sap_info.rigid_volume_elems[i_e][2]
        i_v3 = sap_info.rigid_volume_elems[i_e][3]
        pos_v0 = sap_state.rigid_volume_verts[i_v0, i_b]
        pos_v1 = sap_state.rigid_volume_verts[i_v1, i_b]
        pos_v2 = sap_state.rigid_volume_verts[i_v2, i_b]
        pos_v3 = sap_state.rigid_volume_verts[i_v3, i_b]
        bvh_leaves.aabbs_min[i_b, i_e] = qd.min(pos_v0, pos_v1, pos_v2, pos_v3)
        bvh_leaves.aabbs_max[i_b, i_e] = qd.max(pos_v0, pos_v1, pos_v2, pos_v3)


# ------------------------------------------------------------------------------------
# ------------------------------------ Contacts --------------------------------------
# ------------------------------------------------------------------------------------


@qd.kernel
def kernel_update_contact(
    fem_state: FEMState,
    geoms_state: GeomsState,
    free_verts_state: VertsState,
    fixed_verts_state: VertsState,
    sap_state: SAPState,
    contact_queries_state: SAPContactQueriesState,
    fem_info: FEMInfo,
    links_info: LinksInfo,
    faces_info: FacesInfo,
    verts_info: VertsInfo,
    geoms_info: GeomsInfo,
    sap_info: SAPInfo,
    rigid_config: qd.template(),
    sap_config: qd.template(),
    contact_handlers: qd.template(),
    constraint_handlers: qd.template(),
    errno: qd.Tensor,
) -> bool:
    """Detect the contacts of every handler, and return whether any contact is found.

    A handler past the share of the capacity of an environment drops its new contacts and flags its errors.
    """
    for i_b in range(errno.shape[0]):
        for contact in qd.static(contact_handlers):
            contact.envs_n_contact_pairs[i_b] = 0
            if qd.static(contact.has_candidates):
                contact.envs_n_contact_candidates[i_b] = 0
        if qd.static(sap_config.enable_fem_self_tet_contact):
            contact_queries_state.fem_self.results.trees_count[i_b] = 0
        if qd.static(sap_config.enable_rigid_fem_contact):
            contact_queries_state.rigid_tri.results.trees_count[i_b] = 0
        if qd.static(sap_config.has_rigid_tet_tree):
            contact_queries_state.rigid_tet.results.trees_count[i_b] = 0
    n_contact_pairs = 0
    for contact in qd.static(contact_handlers):
        contact.detection(
            fem_state,
            free_verts_state,
            fixed_verts_state,
            geoms_state,
            sap_state,
            contact_queries_state,
            fem_info,
            links_info,
            verts_info,
            faces_info,
            geoms_info,
            sap_info,
            rigid_config,
            errno,
        )
        n_contact_pairs = n_contact_pairs + contact.n_contact_pairs[None]
    for i_h, handler in qd.static(enumerate(constraint_handlers)):
        sap_state.constraint_order.handlers_n_constraints[i_h + 1] = (
            sap_state.constraint_order.handlers_n_constraints[i_h] + handler.n_constraints[None]
        )
    return n_contact_pairs > 0


@qd.kernel
def kernel_order_constraints(
    sap_state: SAPState,
    fem_info: FEMInfo,
    sap_info: SAPInfo,
    sap_config: qd.template(),
    constraint_handlers: qd.template(),
):
    """
    Sort the constraints of every environment, split them into chunks, and list the contacts of each vertex and link.

    The constraints of each environment sort by their key, and the lists follow that order. The detection gives the
    contacts their slots in the order the threads find them, which changes from a run to another. Every sum of the solve
    runs over the constraints in their sorted order (see func_compute_constraints), and every FEM vertex and rigid link
    gathers its contacts in that order (see func_gather_fem_vertex and func_compute_link_wrench), so that the solve is
    reproducible bitwise. An entry of the list of a vertex packs the rank of its contact with the corner of the contact
    at the vertex, and of a link with the side of the contact on the link. One thread per environment fills its lists.
    """
    # The FEM vertices a contact acts on at most, the corners of its two tets, and the links, its two sides. An entry of
    # the lists of the contacts of each vertex and link packs the rank of the contact in their sorted order with its
    # corner or its side.
    N_CORNERS_MAX = qd.static(8)
    N_SIDES_MAX = qd.static(2)
    # The items of a chunk of the sorted constraints (see CHUNK_SIZE)
    CHUNK_DIM = qd.static(32)
    N_HANDLERS = qd.static(len(constraint_handlers))
    n_envs = sap_state.constraint_order.envs_constraint_start.shape[0] - 1
    key_shift = qd.u64(sap_info.constraints_key_shift[None])
    env_shift = qd.u64(sap_info.constraints_env_shift[None])

    # The constraints of each environment follow those of the environments before it, in chunks
    for _ in range(1):
        i_s = 0
        i_u = 0
        for i_b in range(n_envs):
            sap_state.constraint_order.envs_constraint_start[i_b] = i_s
            sap_state.constraint_order.envs_unit_start[i_b] = i_u
            n_env_constraints = 0
            for handler in qd.static(constraint_handlers):
                n_env_constraints = n_env_constraints + handler.envs_n_constraints(i_b)
            for i_k in range((n_env_constraints + CHUNK_DIM - 1) // CHUNK_DIM):
                sap_state.constraint_order.units_env[i_u] = i_b
                sap_state.constraint_order.units_start[i_u] = i_s + i_k * CHUNK_DIM
                i_u = i_u + 1
            i_s = i_s + n_env_constraints
            if qd.static(gs.backend == gs.cpu):
                sap_state.constraint_order.envs_cursor[i_b] = sap_state.constraint_order.envs_constraint_start[i_b]
        sap_state.constraint_order.envs_constraint_start[n_envs] = i_s
        sap_state.constraint_order.envs_unit_start[n_envs] = i_u
        sap_state.constraint_order.n_units[None] = i_u

    # The key of every constraint, laid out by environment on CPU, then sorted. The device radix sort runs on GPU alone,
    # while on CPU each environment sorts its own constraints by a radix sort of 8 bits per pass.
    sap_state.constraint_order.n_sorted[None] = sap_state.constraint_order.handlers_n_constraints[N_HANDLERS]
    for i_q in range(sap_state.constraint_order.handlers_n_constraints[N_HANDLERS]):
        for i_h, handler in qd.static(enumerate(constraint_handlers)):
            if (
                sap_state.constraint_order.handlers_n_constraints[i_h] <= i_q
                and i_q < sap_state.constraint_order.handlers_n_constraints[i_h + 1]
            ):
                i_c = i_q - sap_state.constraint_order.handlers_n_constraints[i_h]
                i_b = handler.constraint_env(i_c)
                i_s = i_q
                if qd.static(gs.backend == gs.cpu):
                    i_s = qd.atomic_add(sap_state.constraint_order.envs_cursor[i_b], 1)
                sap_state.constraint_order.key[i_s] = (
                    (qd.u64(i_b) << env_shift) | (qd.u64(i_h) << key_shift) | handler.constraint_key(i_c)
                )
                sap_state.constraint_order.order[i_s] = sap_info.handlers_slot_start[i_h] + i_c
    if qd.static(gs.backend == gs.cpu):
        n_passes = (sap_info.constraints_env_shift[None] + 7) // 8
        for i_b in range(n_envs):
            i_s_start = sap_state.constraint_order.envs_constraint_start[i_b]
            i_s_end = sap_state.constraint_order.envs_constraint_start[i_b + 1]
            for i_pass in range(n_passes):
                for i_d in range(256):
                    sap_state.constraint_order.sort_histogram[i_d, i_b] = 0
                for i_s in range(i_s_start, i_s_end):
                    i_d = qd.cast((sap_state.constraint_order.key[i_s] >> qd.u64(8 * i_pass)) & qd.u64(255), gs.qd_int)
                    sap_state.constraint_order.sort_histogram[i_d, i_b] = (
                        sap_state.constraint_order.sort_histogram[i_d, i_b] + 1
                    )
                i_n = i_s_start
                for i_d in range(256):
                    n_digit = sap_state.constraint_order.sort_histogram[i_d, i_b]
                    sap_state.constraint_order.sort_histogram[i_d, i_b] = i_n
                    i_n = i_n + n_digit
                for i_s in range(i_s_start, i_s_end):
                    i_d = qd.cast((sap_state.constraint_order.key[i_s] >> qd.u64(8 * i_pass)) & qd.u64(255), gs.qd_int)
                    i_n = sap_state.constraint_order.sort_histogram[i_d, i_b]
                    sap_state.constraint_order.key_scratch[i_n] = sap_state.constraint_order.key[i_s]
                    sap_state.constraint_order.order_scratch[i_n] = sap_state.constraint_order.order[i_s]
                    sap_state.constraint_order.sort_histogram[i_d, i_b] = i_n + 1
                for i_s in range(i_s_start, i_s_end):
                    sap_state.constraint_order.key[i_s] = sap_state.constraint_order.key_scratch[i_s]
                    sap_state.constraint_order.order[i_s] = sap_state.constraint_order.order_scratch[i_s]
    else:
        radix_sort(
            sap_state.constraint_order.key,
            sap_state.constraint_order.key_scratch,
            sap_state.constraint_order.order,
            sap_state.constraint_order.order_scratch,
            sap_state.constraint_order.sort_scratch,
            sap_state.constraint_order.n_sorted,
            key_dtype=qd.u64,
            has_values=True,
            end_bit=qd.static(sap_config.constraints_sort_end_bit),
            log256_max_n=qd.static(sap_config.constraints_sort_log256),
        )

    # The contacts of each FEM vertex, counted at the next vertex, then summed into the start of the list of each
    # vertex, which the second pass moves to its end while filling it, the last loop moving it back.
    if qd.static(sap_config.has_fem_contacts):
        n_vertices = sap_state.constraint_order.fem_incidence_start.shape[0] - 1
        for i_b in range(n_envs):
            i_n_start = i_b * sap_info.n_env_fem_entries[None]
            for i_v in range(n_vertices + 1):
                sap_state.constraint_order.fem_incidence_start[i_v, i_b] = 0
            for i_pass in range(2):
                if i_pass == 1:
                    i_n = i_n_start
                    for i_v in range(n_vertices + 1):
                        i_n = i_n + sap_state.constraint_order.fem_incidence_start[i_v, i_b]
                        sap_state.constraint_order.fem_incidence_start[i_v, i_b] = i_n
                for i_s in range(
                    sap_state.constraint_order.envs_constraint_start[i_b],
                    sap_state.constraint_order.envs_constraint_start[i_b + 1],
                ):
                    i_q = sap_state.constraint_order.order[i_s]
                    for i_h, handler in qd.static(enumerate(constraint_handlers)):
                        if qd.static(handler.n_fem_corners > 0):
                            if sap_info.handlers_slot_start[i_h] <= i_q and i_q < sap_info.handlers_slot_start[i_h + 1]:
                                i_c = i_q - sap_info.handlers_slot_start[i_h]
                                for i_corner in range(handler.n_fem_corners):
                                    i_v = handler.fem_corner_vertex(i_c, i_corner, fem_info)
                                    if i_pass == 0:
                                        sap_state.constraint_order.fem_incidence_start[i_v + 1, i_b] = (
                                            sap_state.constraint_order.fem_incidence_start[i_v + 1, i_b] + 1
                                        )
                                    else:
                                        i_n = sap_state.constraint_order.fem_incidence_start[i_v, i_b]
                                        sap_state.constraint_order.fem_incidence[i_n] = i_s * N_CORNERS_MAX + i_corner
                                        sap_state.constraint_order.fem_incidence_start[i_v, i_b] = i_n + 1
            for i_v_ in range(n_vertices):
                i_v = n_vertices - i_v_
                sap_state.constraint_order.fem_incidence_start[i_v, i_b] = (
                    sap_state.constraint_order.fem_incidence_start[i_v - 1, i_b]
                )
            sap_state.constraint_order.fem_incidence_start[0, i_b] = i_n_start

    # The contacts of each rigid link, listed the same way
    if qd.static(sap_config.has_rigid_contacts):
        n_links = sap_state.constraint_order.link_incidence_start.shape[0] - 1
        for i_b in range(n_envs):
            i_n_start = i_b * sap_info.n_env_link_entries[None]
            for i_l in range(n_links + 1):
                sap_state.constraint_order.link_incidence_start[i_l, i_b] = 0
            for i_pass in range(2):
                if i_pass == 1:
                    i_n = i_n_start
                    for i_l in range(n_links + 1):
                        i_n = i_n + sap_state.constraint_order.link_incidence_start[i_l, i_b]
                        sap_state.constraint_order.link_incidence_start[i_l, i_b] = i_n
                for i_s in range(
                    sap_state.constraint_order.envs_constraint_start[i_b],
                    sap_state.constraint_order.envs_constraint_start[i_b + 1],
                ):
                    i_q = sap_state.constraint_order.order[i_s]
                    for i_h, handler in qd.static(enumerate(constraint_handlers)):
                        if qd.static(handler.n_rigid_sides > 0):
                            if sap_info.handlers_slot_start[i_h] <= i_q and i_q < sap_info.handlers_slot_start[i_h + 1]:
                                i_c = i_q - sap_info.handlers_slot_start[i_h]
                                for i_side in range(handler.n_rigid_sides):
                                    i_l = handler.rigid_side_link(i_c, i_side)
                                    if i_pass == 0:
                                        sap_state.constraint_order.link_incidence_start[i_l + 1, i_b] = (
                                            sap_state.constraint_order.link_incidence_start[i_l + 1, i_b] + 1
                                        )
                                    else:
                                        i_n = sap_state.constraint_order.link_incidence_start[i_l, i_b]
                                        sap_state.constraint_order.link_incidence[i_n] = i_s * N_SIDES_MAX + i_side
                                        sap_state.constraint_order.link_incidence_start[i_l, i_b] = i_n + 1
            for i_l_ in range(n_links):
                i_l = n_links - i_l_
                sap_state.constraint_order.link_incidence_start[i_l, i_b] = (
                    sap_state.constraint_order.link_incidence_start[i_l - 1, i_b]
                )
            sap_state.constraint_order.link_incidence_start[0, i_b] = i_n_start


@qd.kernel
def kernel_build_equality_constraints(
    equalities_info: EqualitiesInfo,
    joints_info: JointsInfo,
    sap_info: SAPInfo,
    rigid_config: qd.template(),
    constraint_handler: qd.template(),
):
    """Build the joint equality constraints of every environment, the same in all of them.

    The joint equalities of environment i_b take the slots from i_b times their number, so that their order is the same
    on every run.
    """
    constraints = qd.static(constraint_handler.constraints)
    n_env_constraints = constraint_handler.joint_equalities_idx.shape[0]
    for i_c in range(constraints.batch_idx.shape[0]):
        for i_d in range(constraint_handler.Jt.shape[1]):
            constraint_handler.Jt[i_c, i_d] = qd.Vector([0.0])
    for i_b, i_k in qd.ndrange(constraints.batch_idx.shape[0] // n_env_constraints, n_env_constraints):
        i_e = constraint_handler.joint_equalities_idx[i_k]
        i_c = i_b * n_env_constraints + i_k
        constraints[i_c].batch_idx = i_b
        i_j1 = equalities_info.eq_obj1id[i_e, i_b]
        i_j2 = equalities_info.eq_obj2id[i_e, i_b]
        I_j1 = [i_j1, i_b] if qd.static(rigid_config.batch_joints_info) else i_j1
        I_j2 = [i_j2, i_b] if qd.static(rigid_config.batch_joints_info) else i_j2
        i_dof1 = joints_info.dof_start[I_j1]
        constraints[i_c].i_eq = i_e
        constraints[i_c].i_dof1 = i_dof1
        constraints[i_c].i_dof2 = joints_info.dof_start[I_j2]
        constraints[i_c].i_qpos1 = joints_info.q_start[I_j1]
        constraints[i_c].i_qpos2 = joints_info.q_start[I_j2]
        constraints[i_c].constraint_info.k = constraint_handler.stiffness[None]
        constraints[i_c].constraint_info.R_inv = sap_info.dt2[None] * constraint_handler.stiffness[None]
        constraints[i_c].constraint_info.R = 1.0 / constraints[i_c].constraint_info.R_inv
        constraints[i_c].constraint_info.v_hat = 0.0
        constraint_handler.Jt[i_c, i_dof1] = qd.Vector([1.0])


# ------------------------------------------------------------------------------------
# ------------------------------------- Solve ----------------------------------------
# ------------------------------------------------------------------------------------


@qd.kernel
def kernel_compute_regularization(
    fem_state: FEMState,
    dofs_state: DofsState,
    links_state: LinksState,
    sap_state: SAPState,
    fem_info: FEMInfo,
    entities_info: EntitiesInfo,
    links_info: LinksInfo,
    equalities_info: EqualitiesInfo,
    rigid_info: RigidInfo,
    sap_info: SAPInfo,
    rigid_config: qd.template(),
    sap_config: qd.template(),
    contact_handlers: qd.template(),
    constraint_handler: qd.template(),
):

    for contact in qd.static(contact_handlers):
        contact.compute_regularization(
            fem_state, dofs_state, links_state, sap_state, fem_info, links_info, rigid_info, sap_info, sap_config
        )
    if qd.static(sap_config.has_equality_constraints):
        constraint_handler.compute_regularization(entities_info, equalities_info, rigid_info, sap_info)


@qd.kernel
def kernel_update_vel(
    fem_state: FEMState,
    dofs_state: DofsState,
    sap_state: SAPState,
    sap_info: SAPInfo,
    sap_config: qd.template(),
    contact_handlers: qd.template(),
    errno: qd.Tensor,
):
    """
    Write the velocities of the solve back to the solvers, flagging the environments whose solve stopped at its Newton
    iteration cap with a residual above CAP_RESIDUAL_MAX.

    The environments still active at the end of the solve are those whose last convergence check failed.
    """
    for i_b in range(errno.shape[0]):
        if sap_state.envs.is_active[i_b] and sap_state.envs.relative_residual[i_b] >= sap_info.cap_residual_max[None]:
            errno[i_b] = errno[i_b] | SAPErrorCode.INVALID_SOLVE
    if qd.static(sap_config.has_fem):
        for i_v, i_b in qd.ndrange(*sap_state.fem_vertices.v.shape, axes=qd.static(eoa(fem_state.vertices.vel))):
            fem_state.vertices.vel_next[i_v, i_b] = sap_state.fem_vertices.v[i_v, i_b]
    if qd.static(sap_config.has_rigid):
        for i_d, i_b in qd.ndrange(sap_info.n_dofs[None], errno.shape[0], axes=qd.static(eoa(dofs_state.vel))):
            dofs_state.vel[i_d, i_b] = sap_state.rigid_dofs.v[i_d, i_b]


@qd.kernel
def kernel_init_sap_solve(
    newton_counter: qd.types.ndarray(qd.i32, ndim=0),
    fem_state: FEMState,
    dofs_state: DofsState,
    sap_state: SAPState,
    fem_info: FEMInfo,
    vertex_constraints: FEMVertexConstraints,
    sap_info: SAPInfo,
    fem_config: qd.template(),
    sap_config: qd.template(),
):
    """
    Start the SAP solve of every environment from the start-of-substep velocities, with its Newton iterations left.

    The entities in contact start from their velocities at the start of the substep, the solution of the previous one,
    which the contacts of a resting or slowly changing scene leave close to the solution of this one, where the free
    velocities are far from it. The other entities start from their free velocities, which solve their share of the
    cost exactly. The constrained FEM vertices, which the solve leaves as they start, start from their free velocities,
    which follow their constraints.
    """
    N_LANES = qd.static(32 if sap_config.enable_cooperative_kernels else 1)
    newton_counter[()] = sap_info.n_newton_iterations[None]
    if qd.static(sap_config.has_fem):
        # The velocities of the FEM vertices of each entity, which are also the reference of its cost. A free vertex of
        # an entity in contact starts from its velocity at the start of the substep, and of an entity without contact
        # from the inertia guess, which gravity alone moves. A constrained vertex starts from the velocity that follows
        # its constraint.
        for i_e, i_b in qd.ndrange(sap_state.fem_coarse.mass.shape[0], sap_state.fem_coarse.mass.shape[1]):
            is_in_contact = False
            for i_k in range(sap_info.entities_chunk_start[i_e], sap_info.entities_chunk_start[i_e + 1]):
                for i_v in range(sap_info.chunks_vertex_start[i_k], sap_info.chunks_vertex_end[i_k]):
                    if qd.static(sap_config.has_fem_contacts):
                        if (
                            sap_state.constraint_order.fem_incidence_start[i_v + 1, i_b]
                            > sap_state.constraint_order.fem_incidence_start[i_v, i_b]
                        ):
                            is_in_contact = True
            for i_k in range(sap_info.entities_chunk_start[i_e], sap_info.entities_chunk_start[i_e + 1]):
                for i_v in range(sap_info.chunks_vertex_start[i_k], sap_info.chunks_vertex_end[i_k]):
                    vel = fem_state.vertices.vel_next[i_v, i_b]
                    if is_in_contact and func_is_fem_vertex_free(i_v, i_b, vertex_constraints, sap_config):
                        vel = fem_state.vertices.vel[i_v, i_b]
                    sap_state.fem_vertices.v[i_v, i_b] = vel
    if qd.static(sap_config.has_rigid):
        # The rigid dofs of each kinematic tree, in contact when the contact lists of the links of the subtree of its
        # root hold any contact.
        for i_d, i_b in qd.ndrange(
            sap_info.n_dofs[None], sap_state.envs.is_active.shape[0], axes=qd.static(eoa(sap_state.rigid_dofs.v))
        ):
            is_in_contact = False
            if qd.static(sap_config.has_rigid_contacts):
                i_r = sap_info.dofs_root_link_idx[i_d]
                i_l_end = sap_info.links_subtree_end[i_r]
                is_in_contact = (
                    sap_state.constraint_order.link_incidence_start[i_l_end, i_b]
                    > sap_state.constraint_order.link_incidence_start[i_r, i_b]
                )
            v = dofs_state.vel[i_d, i_b]
            if is_in_contact:
                v = dofs_state.vel_prev[i_d, i_b]
            sap_state.rigid_dofs.v[i_d, i_b] = v
    for i_b in range(sap_state.envs.is_active.shape[0]):
        sap_state.envs.is_active[i_b] = True
        sap_state.envs.energy_newton_prev[i_b] = qd.math.inf
    if qd.static(sap_config.has_fem):
        # The inertia of the rigid motions of each FEM entity, which their Hessian starts from at every Newton
        # iteration. The rigid motion (c_lin, c_ang) moves a vertex at c_lin + c_ang x d, d its position relative to
        # x_ref, whose inertia sums a d^T d terms over the vertices that move, a being the inertia of a vertex in the
        # cost. The lanes of a chunk take an environment and an entity, striding its vertices.
        inertia_factor = fem_info.dt2[None] * (fem_info.damping_alpha_dt[None] + 1.0)
        for i_t in range(sap_state.fem_coarse.mass.shape[1] * sap_state.fem_coarse.mass.shape[0] * N_LANES):
            n_entities = sap_state.fem_coarse.mass.shape[0]
            i_lane = i_t % N_LANES
            i_e = i_t // N_LANES % n_entities
            i_b = i_t // (N_LANES * n_entities)
            i_k_start = sap_info.entities_chunk_start[i_e]
            i_k_last = sap_info.entities_chunk_start[i_e + 1] - 1
            i_v_start = sap_info.chunks_vertex_start[i_k_start]
            i_v_end = sap_info.chunks_vertex_end[i_k_last]
            x_ref = fem_state.vertices.pos[i_v_start, i_b]
            # The mass, then its first and second moments
            mass = gs.qd_float(0.0)
            mass_first = qd.Vector.zero(gs.qd_float, 3)
            mass_second = qd.Matrix.zero(gs.qd_float, 3, 3)
            for i_v_ in range((i_v_end - i_v_start + N_LANES - 1) // N_LANES):
                i_v = i_v_start + i_v_ * N_LANES + i_lane
                if i_v < i_v_end:
                    if func_is_fem_vertex_free(i_v, i_b, vertex_constraints, sap_config):
                        a = fem_info.vertices.mass_over_dt2[i_v] * inertia_factor
                        d = fem_state.vertices.pos[i_v, i_b] - x_ref
                        mass = mass + a
                        mass_first = mass_first + a * d
                        mass_second = mass_second + a * (
                            d.norm_sqr() * qd.Matrix.identity(gs.qd_float, 3) - d.outer_product(d)
                        )
            if qd.static(N_LANES > 1):
                # The lanes sum scalars alone
                mass = su.qd_block_sum(mass)
                for k in qd.static(range(3)):
                    mass_first[k] = su.qd_block_sum(mass_first[k])
                    for l in qd.static(range(3)):
                        mass_second[k, l] = su.qd_block_sum(mass_second[k, l])
            if i_lane == 0:
                sap_state.fem_coarse.x_ref[i_e, i_b] = x_ref
                sap_state.fem_coarse.mass[i_e, i_b] = mass
                sap_state.fem_coarse.mass_first[i_e, i_b] = mass_first
                sap_state.fem_coarse.mass_second[i_e, i_b] = mass_second


@qd.func
def func_is_fem_vertex_free(
    i_v: int, i_b: int, vertex_constraints: FEMVertexConstraints, sap_config: qd.template()
) -> bool:
    """Whether a FEM vertex moves in the solve, unconstrained.

    The vertex is i_v of the environment i_b.
    """
    is_free = True
    if qd.static(sap_config.enable_vertex_constraints):
        is_free = not vertex_constraints.is_constrained[i_v, i_b]
    return is_free


# ------------------------------------------------------------------------------------
# ------------------------------ Fixed-order accumulation ------------------------------
# ------------------------------------------------------------------------------------


@qd.func
def func_chunks_total(
    i_b: int,
    i_k_start: int,
    i_k_end: int,
    i_q: int,
    i_lane: int,
    sap_state: SAPState,
    n_lanes: qd.template(),
    n_sums: qd.template(),
):
    """
    Return sums over a range of chunks of an environment, each added up over the chunks in order.

    The `n_sums` sums from i_q of the environment i_b run over the chunks [i_k_start, i_k_end) of the vertices then the
    dofs (see func_compute_vertices_dofs). With `n_lanes` above one, every lane of a chunk calls it with the same
    arguments but its own lane i_lane, and gets the same bits: the lanes stride the chunks, then sum their shares along
    a fixed tree. A single thread sums them in a row.
    """
    total = qd.Vector.zero(gs.qd_float, n_sums)
    for i_k_ in range((i_k_end - i_k_start + n_lanes - 1) // n_lanes):
        i_k = i_k_start + i_k_ * n_lanes + i_lane
        if i_k < i_k_end:
            for j in qd.static(range(n_sums)):
                total[j] = total[j] + sap_state.vertices_dofs_partials[i_k, i_q + j, i_b]
    if qd.static(n_lanes > 1):
        for j in qd.static(range(n_sums)):
            total[j] = su.qd_block_sum(total[j])
    return total


@qd.func
def func_constraints_total(i_b: int, i_q: int, sap_state: SAPState):
    """Return a sum of the constraints of an environment, added up over their chunks in order."""
    total = gs.qd_float(0.0)
    for i_u in range(
        sap_state.constraint_order.envs_unit_start[i_b], sap_state.constraint_order.envs_unit_start[i_b + 1]
    ):
        total = total + sap_state.constraints_partials[i_u, i_q]
    return total


@qd.func
def func_gather_fem_vertex(
    i_t: int,
    envs_is_active: qd.Tensor,
    sap_state: SAPState,
    fem_info: FEMInfo,
    vertex_constraints: FEMVertexConstraints,
    sap_info: SAPInfo,
    sap_config: qd.template(),
    constraint_handlers: qd.template(),
    op: qd.template(),
):
    """
    Add the elastic forces and the contact terms of a FEM vertex to it, in a fixed order.

    The elastic forces are those of the elements at the corners of the vertex (see func_compute_fem_corner_force), and
    the contact terms J^T x and, for the gradient operation `op`, J^T G J, summed over the contacts at the vertex in
    their sorted order. x is the impulse of each contact for the gradient, and the product of its Hessian with the
    Jacobian of the search direction for the conjugate gradient. The thread i_t takes a lane of a vertex of an
    environment. On GPU the lanes of a tile stride the corners then the contacts of a vertex and sum their shares along
    a fixed tree, on CPU one thread takes them in a row. The gradient takes the elastic forces and the contact impulses,
    the product with the search direction both, and the product with the Newton step of the line search the elastic
    forces alone.
    """
    # The corners an entry packs (see kernel_order_constraints), and the lanes of a GPU subgroup tile that gather
    # together the element corners and the contacts of a vertex.
    N_CORNERS_MAX = qd.static(8)
    LOG2_GATHER_LANES = qd.static(3)
    N_LANES = qd.static(2**LOG2_GATHER_LANES if sap_config.enable_cooperative_kernels else 1)
    n_vertices = sap_state.fem_vertices.v.shape[0]
    i_lane = i_t % N_LANES
    i_v = i_t // N_LANES % n_vertices
    i_b = i_t // (N_LANES * n_vertices)
    elastic = qd.Vector.zero(gs.qd_float, 3)
    Jt_x = qd.Vector.zero(gs.qd_float, 3)
    JtGJ = qd.Matrix.zero(gs.qd_float, 3, 3)
    if envs_is_active[i_b]:
        i_corner_start = fem_info.vertices_corners_start[i_v]
        n_corners = fem_info.vertices_corners_start[i_v + 1] - i_corner_start
        n_contacts = 0
        i_n_start = 0
        if qd.static(sap_config.has_fem_contacts and op != VertexDofOp.LINESEARCH_D2ELLA):
            i_n_start = sap_state.constraint_order.fem_incidence_start[i_v, i_b]
            if func_is_fem_vertex_free(i_v, i_b, vertex_constraints, sap_config):
                n_contacts = sap_state.constraint_order.fem_incidence_start[i_v + 1, i_b] - i_n_start
        for i_k_ in range((n_corners + n_contacts + N_LANES - 1) // N_LANES):
            i_k = i_k_ * N_LANES + i_lane
            if i_k < n_corners:
                elastic = elastic + sap_state.fem_corner_forces[i_corner_start + i_k, i_b]
            elif i_k < n_corners + n_contacts:
                i_entry = sap_state.constraint_order.fem_incidence[i_n_start + i_k - n_corners]
                i_q = sap_state.constraint_order.order[i_entry // N_CORNERS_MAX]
                i_corner = i_entry % N_CORNERS_MAX
                for i_h, handler in qd.static(enumerate(constraint_handlers)):
                    if qd.static(handler.n_fem_corners > 0):
                        if sap_info.handlers_slot_start[i_h] <= i_q and i_q < sap_info.handlers_slot_start[i_h + 1]:
                            i_c = i_q - sap_info.handlers_slot_start[i_h]
                            contact_info = qd.static(handler.contact_pairs.contact_info)
                            w = handler.fem_corner_weight(i_c, i_corner)
                            frame = handler.contact_frame(i_c)
                            if qd.static(op == VertexDofOp.GRADIENT):
                                Jt_x = Jt_x + w * (frame @ contact_info[i_c].gamma)
                                JtGJ = JtGJ + w**2 * (frame @ contact_info[i_c].G @ frame.transpose())
                            else:
                                Jt_x = Jt_x + w * (frame @ contact_info[i_c].GJp)
    if qd.static(N_LANES > 1):
        for i in qd.static(range(3)):
            elastic[i] = su.qd_block_sum(elastic[i], LOG2_GATHER_LANES)
            if qd.static(op != VertexDofOp.LINESEARCH_D2ELLA):
                Jt_x[i] = su.qd_block_sum(Jt_x[i], LOG2_GATHER_LANES)
            if qd.static(op == VertexDofOp.GRADIENT):
                for j in qd.static(range(3)):
                    JtGJ[i, j] = su.qd_block_sum(JtGJ[i, j], LOG2_GATHER_LANES)
    if i_lane == 0 and envs_is_active[i_b]:
        if qd.static(op == VertexDofOp.GRADIENT):
            sap_state.fem_vertices.gradient[i_v, i_b] = sap_state.fem_vertices.gradient[i_v, i_b] + elastic - Jt_x
            sap_state.fem_vertices.impulse[i_v, i_b] = Jt_x
            diag3x3 = sap_state.fem_vertices.diag3x3[i_v, i_b] + JtGJ
            sap_state.fem_vertices.diag3x3[i_v, i_b] = diag3x3
            sap_state.fem_vertices.prec[i_v, i_b] = diag3x3.inverse()
        if qd.static(op == VertexDofOp.PCG_PTAP):
            sap_state.fem_vertices.Ap[i_v, i_b] = sap_state.fem_vertices.Ap[i_v, i_b] + elastic + Jt_x
        if qd.static(op == VertexDofOp.LINESEARCH_D2ELLA):
            sap_state.fem_vertices.dp[i_v, i_b] = sap_state.fem_vertices.dp[i_v, i_b] + elastic


@qd.func
def func_compute_link_wrench(
    i_t: int,
    envs_is_active: qd.Tensor,
    links_state: LinksState,
    sap_state: SAPState,
    sap_info: SAPInfo,
    sap_config: qd.template(),
    constraint_handlers: qd.template(),
    op: qd.template(),
):
    """
    Sum the wrench that the contacts of a link apply to it, in the sorted order of the contacts.

    The thread i_t takes a lane of a share of a link of an environment. The wrench holds the force and the torque about
    the root of the kinematic tree. The shares of a link (see N_LINK_WRENCH_SHARES) take its contacts in turn by chunks
    (see CHUNK_SIZE), the lanes of each summing theirs along a fixed tree into one wrench per share. The force of a
    contact is its impulse for the gradient (op VertexDofOp.GRADIENT), and the product of its Hessian with the Jacobian
    of the search direction for the conjugate gradient (op VertexDofOp.PCG_PTAP). The dofs add up the wrenches of the
    links of their subtree (see func_gather_rigid_dof).
    """
    # The sides an entry packs (see kernel_order_constraints)
    N_SIDES_MAX = qd.static(2)
    N_LANES = qd.static(32 if sap_config.enable_cooperative_kernels else 1)
    N_SHARES = qd.static(2 if sap_config.enable_cooperative_kernels else 1)
    n_links = sap_state.constraint_order.links_wrench_force.shape[0] // N_SHARES
    i_lane = i_t % N_LANES
    i_w = i_t // N_LANES % N_SHARES
    i_l = i_t // (N_LANES * N_SHARES) % n_links
    i_b = i_t // (N_LANES * N_SHARES * n_links)
    wrench = qd.Vector.zero(gs.qd_float, 6)
    if envs_is_active[i_b]:
        i_n_start = sap_state.constraint_order.link_incidence_start[i_l, i_b]
        i_n_end = sap_state.constraint_order.link_incidence_start[i_l + 1, i_b]
        for i_n_ in range((i_n_end - i_n_start + N_SHARES * N_LANES - 1) // (N_SHARES * N_LANES)):
            i_n = i_n_start + (i_n_ * N_SHARES + i_w) * N_LANES + i_lane
            if i_n < i_n_end:
                i_entry = sap_state.constraint_order.link_incidence[i_n]
                i_q = sap_state.constraint_order.order[i_entry // N_SIDES_MAX]
                i_side = i_entry % N_SIDES_MAX
                for i_h, handler in qd.static(enumerate(constraint_handlers)):
                    if qd.static(handler.n_rigid_sides > 0):
                        if sap_info.handlers_slot_start[i_h] <= i_q and i_q < sap_info.handlers_slot_start[i_h + 1]:
                            i_c = i_q - sap_info.handlers_slot_start[i_h]
                            contact_info = qd.static(handler.contact_pairs.contact_info)
                            x = contact_info[i_c].gamma
                            if qd.static(op == VertexDofOp.PCG_PTAP):
                                x = contact_info[i_c].GJp
                            f = handler.rigid_side_sign(i_side) * (handler.contact_frame(i_c) @ x)
                            t_pos = handler.contact_pairs[i_c].contact_pos - links_state.root_COM[i_l, i_b]
                            wrench[:3] = wrench[:3] + f
                            wrench[3:] = wrench[3:] + t_pos.cross(f)
    if qd.static(N_LANES > 1):
        for k in qd.static(range(6)):
            wrench[k] = su.qd_block_sum(wrench[k])
    if i_lane == 0 and envs_is_active[i_b]:
        sap_state.constraint_order.links_wrench_force[i_l * N_SHARES + i_w, i_b] = qd.Vector(
            [wrench[0], wrench[1], wrench[2]]
        )
        sap_state.constraint_order.links_wrench_torque[i_l * N_SHARES + i_w, i_b] = qd.Vector(
            [wrench[3], wrench[4], wrench[5]]
        )


@qd.func
def func_gather_rigid_dof(
    i_d: int,
    i_b: int,
    dofs_state: DofsState,
    sap_state: SAPState,
    sap_info: SAPInfo,
    sap_config: qd.template(),
    constraint_handler: qd.template(),
    op: qd.template(),
):
    """
    Return J^T x at a rigid dof, summed over the constraints acting on it in a fixed order.

    The contacts act on the dof i_d of the environment i_b through the wrench shares of the links of the subtree of the
    dof, in order (see func_compute_link_wrench), then the joint equalities on it.
    """
    N_SHARES = qd.static(2 if sap_config.enable_cooperative_kernels else 1)
    Jt_x = gs.qd_float(0.0)
    if qd.static(sap_config.has_rigid_contacts):
        i_l = sap_info.dofs_link_idx[i_d]
        for i_w in range(i_l * N_SHARES, sap_info.links_subtree_end[i_l] * N_SHARES):
            Jt_x = Jt_x + dofs_state.cdof_vel[i_d, i_b].dot(sap_state.constraint_order.links_wrench_force[i_w, i_b])
            Jt_x = Jt_x + dofs_state.cdof_ang[i_d, i_b].dot(sap_state.constraint_order.links_wrench_torque[i_w, i_b])
    if qd.static(sap_config.has_equality_constraints):
        constraints = qd.static(constraint_handler.constraints)
        n_equalities = constraint_handler.joint_equalities_idx.shape[0]
        for i_k in range(n_equalities):
            i_c = i_b * n_equalities + i_k
            x = constraints[i_c].constraint_info.gamma
            if qd.static(op == VertexDofOp.PCG_PTAP):
                x = constraints[i_c].constraint_info.GJp
            if constraints[i_c].i_dof1 == i_d:
                Jt_x = Jt_x + x
            if constraints[i_c].i_dof2 == i_d:
                Jt_x = Jt_x + constraints[i_c].jac_dof2 * x
    return Jt_x


@qd.func
def func_update_envs_active(
    newton_counter: qd.Tensor,
    sap_state: SAPState,
    fem_info: FEMInfo,
    sap_info: SAPInfo,
    sap_config: qd.template(),
    constraint_handlers: qd.template(),
):
    """
    Stop the environments whose gradient is small enough, the others adding up the coarse Hessians of the contacts.

    The lanes of a chunk (see CHUNK_SIZE) take an environment, adding up its norms and the Hessians of its contacts
    restricted to the rigid motions of each FEM entity, which they stride in their sorted order.
    """
    N_LANES = qd.static(32 if sap_config.enable_cooperative_kernels else 1)
    for i_t in range(sap_state.envs.is_active.shape[0] * N_LANES):
        i_lane = i_t % N_LANES
        i_b = i_t // N_LANES
        is_active = sap_state.envs.is_active[i_b]
        if qd.static(N_LANES > 1):
            # The first lane overwrites the flag below, which nothing orders after the reads of the other lanes, so they
            # take the flag it read: lanes disagreeing on it would sum the coarse Hessians in divergent flow.
            is_active = su.qd_block_shuffle(qd.cast(is_active, qd.i32), 0) != 0
        norms = func_chunks_total(
            i_b,
            0,
            sap_state.vertices_dofs_partials.shape[0],
            VertexDofSum.GRADIENT_NORM,
            i_lane,
            sap_state,
            n_lanes=N_LANES,
            n_sums=3,
        )
        gradient_norm = norms[VertexDofSum.GRADIENT_NORM]
        momentum_norm = norms[VertexDofSum.MOMENTUM_NORM]
        impulse_norm = norms[VertexDofSum.IMPULSE_NORM]
        if is_active:
            # The sums are squared norms, compared as norms like the tolerances. The tolerance scales with the momentum
            # alone: the impulses holding a resting body against gravity would otherwise dominate it, leaving the slow
            # motions of that body under the threshold so that they never damp out. The residual checked at the
            # iteration cap keeps the impulses in its scale, which a body at rest has alone.
            norm_thr = sap_info.convergence_atol[None] + sap_info.convergence_rtol[None] * qd.sqrt(momentum_norm)
            is_active = qd.sqrt(gradient_norm) >= norm_thr
            if i_lane == 0:
                sap_state.envs.is_active[i_b] = is_active
                sap_state.envs.relative_residual[i_b] = qd.sqrt(
                    gradient_norm / qd.max(qd.max(momentum_norm, impulse_norm), sap_info.precision[None])
                )
                if is_active:
                    newton_counter[()] = sap_state.loops.n_newton_next[()]
        if qd.static(sap_config.has_fem):
            if is_active:
                i_s_start = sap_state.constraint_order.envs_constraint_start[i_b]
                i_s_end = sap_state.constraint_order.envs_constraint_start[i_b + 1]
                for i_e in range(sap_state.fem_coarse.mass.shape[0]):
                    A = qd.Matrix.zero(gs.qd_float, 6, 6)
                    for i_s_ in range((i_s_end - i_s_start + N_LANES - 1) // N_LANES):
                        i_s = i_s_start + i_s_ * N_LANES + i_lane
                        if i_s < i_s_end:
                            i_q = sap_state.constraint_order.order[i_s]
                            for i_h, handler in qd.static(enumerate(constraint_handlers)):
                                if qd.static(handler.n_fem_corners > 0):
                                    if (
                                        sap_info.handlers_slot_start[i_h] <= i_q
                                        and i_q < sap_info.handlers_slot_start[i_h + 1]
                                    ):
                                        i_c = i_q - sap_info.handlers_slot_start[i_h]
                                        for i_side in range(handler.n_coarse_sides):
                                            i_e_c, H = handler.coarse_hessian(
                                                i_c, i_side, sap_state, fem_info, sap_info
                                            )
                                            if i_e_c == i_e:
                                                A = A + H
                    if qd.static(N_LANES > 1):
                        for k, l in qd.static(qd.ndrange(6, 6)):
                            A[k, l] = su.qd_block_sum(A[k, l])
                    if i_lane == 0:
                        # The inertia of the rigid motions, from the moments of the vertices about x_ref
                        mass = sap_state.fem_coarse.mass[i_e, i_b]
                        first = gu.qd_vec_to_skew(sap_state.fem_coarse.mass_first[i_e, i_b])
                        A[:3, :3] = A[:3, :3] + mass * qd.Matrix.identity(gs.qd_float, 3)
                        A[:3, 3:] = A[:3, 3:] - first
                        A[3:, :3] = A[3:, :3] + first
                        A[3:, 3:] = A[3:, 3:] + sap_state.fem_coarse.mass_second[i_e, i_b]
                        sap_state.fem_coarse.A_inv[i_e, i_b] = func_inverse_spd6(A)


@qd.func
def func_fem_coarse_contact_hessian(
    i_v: int,
    i_b: int,
    contact_pos: qd.types.vector(3),
    frame: qd.types.matrix(3, 3),
    G: qd.types.matrix(3, 3),
    sap_state: SAPState,
    sap_info: SAPInfo,
):
    """
    Return a FEM entity and the Hessian of one of its contacts restricted to its rigid motions.

    The entity is the one of the vertex i_v of the environment i_b. `frame` holds the axes of the contact frame in its
    columns, and `G` is the Hessian of the contact in that frame. The rigid motion (c_lin, c_ang) moves the contact
    point at c_lin + c_ang x d, d its position relative to x_ref.
    """
    i_e = sap_info.fem_vertices_entity_idx[i_v]
    S = gu.qd_vec_to_skew(contact_pos - sap_state.fem_coarse.x_ref[i_e, i_b])
    # The contact velocity in the contact frame is J (c_lin, c_ang) with J = [F^T, -F^T S], F the frame, so that J^T G J
    # holds the blocks of the Hessian of the contact in world axes, S^T being -S.
    G_world = frame @ G @ frame.transpose()
    G_world_S = G_world @ S
    H = qd.Matrix.zero(gs.qd_float, 6, 6)
    H[:3, :3] = G_world
    H[:3, 3:] = -G_world_S
    H[3:, :3] = S @ G_world
    H[3:, 3:] = -S @ G_world_S
    return i_e, H


@qd.func
def func_compute_fem_matrix_vector_product(
    src: qd.Tensor,
    dst: qd.Tensor,
    envs_is_active: qd.Tensor,
    fem_state: FEMState,
    sap_state: SAPState,
    fem_info: FEMInfo,
    vertex_constraints: FEMVertexConstraints,
    sap_info: SAPInfo,
    fem_config: qd.template(),
):
    """
    Compute the inertial part of a FEM Hessian product with a vector, and the corner forces of the elements.

    The product of `src` goes to `dst`, and the vertices add the corner forces (see func_gather_fem_vertex).
    """
    inertia_factor = fem_info.dt2[None] * (fem_info.damping_alpha_dt[None] + 1.0)
    for i_v, i_b in qd.ndrange(*dst.shape, axes=qd.static(eoa(dst))):
        if envs_is_active[i_b]:
            dst[i_v, i_b] = fem_info.vertices.mass_over_dt2[i_v] * src[i_v, i_b] * inertia_factor
    for i_e, i_b in qd.ndrange(fem_info.elements.V.shape[0], dst.shape[1], axes=qd.static(eoa(dst))):
        if envs_is_active[i_b]:
            func_compute_fem_corner_force(
                i_e, i_b, src, fem_state, sap_state, fem_info, vertex_constraints, sap_info, fem_config
            )


@qd.func
def func_compute_fem_corner_force(
    i_e: int,
    i_b: int,
    src: qd.Tensor,
    fem_state: FEMState,
    sap_state: SAPState,
    fem_info: FEMInfo,
    vertex_constraints: FEMVertexConstraints,
    sap_info: SAPInfo,
    fem_config: qd.template(),
):
    """
    Compute the force of a FEM element at each of its corners along a displacement of the vertices.

    The displacement is `src`, and the element is i_e of the environment i_b. The forces are the product of its elastic
    Hessian with the displacement, computed once per element, which func_gather_fem_vertex sums at each vertex.
    """
    i_vs = fem_info.elements.el2v[i_e]
    S = func_element_mapping_matrix(i_e, i_b, fem_info, vertex_constraints, fem_config)
    x = qd.Matrix.cols([src[i_vs[i], i_b] for i in qd.static(range(4))])
    _, dP = func_elastic_product(i_e, i_b, S, x, fem_state, fem_info, fem_config)
    forces = fem_info.elements.V[i_e] * fem_info.dt2[None] * fem_info.damping_beta_factor[None] * dP @ S.transpose()
    for k in qd.static(range(4)):
        i_rank = sap_info.fem_corners_rank[4 * i_e + k]
        sap_state.fem_corner_forces[i_rank, i_b] = forces[:, k]


@qd.func
def func_compute_rigid_mass_mat_vec_product(
    vec: qd.Tensor, out: qd.Tensor, envs_is_active: qd.Tensor, rigid_info: RigidInfo, sap_info: SAPInfo
):
    """Compute the product of the rigid mass matrix with a vector of the dofs, for the active environments."""
    for i_d1, i_b in qd.ndrange(sap_info.n_dofs[None], out.shape[1], axes=qd.static(eoa(rigid_info.mass_mat))):
        if envs_is_active[i_b]:
            prod = gs.qd_float(0.0)
            for i_d0 in range(sap_info.n_dofs[None]):
                prod = prod + rigid_info.mass_mat[i_d1, i_d0, i_b] * vec[i_d0, i_b]
            out[i_d1, i_b] = prod


@qd.func
def func_link_point_jacobian_product(
    i_l: int,
    i_b: int,
    pos: qd.types.vector(3),
    x: qd.Tensor,
    dofs_state: DofsState,
    links_state: LinksState,
    links_info: LinksInfo,
    sap_config: qd.template(),
):
    """
    Return the product of the Jacobian of the velocity of a point attached to a link with a vector of the rigid dofs.

    The point is `pos`, the link i_l and the vector `x`. The Jacobian J is nonzero at the dofs of the link and of its
    ancestors alone, which this func walks up to the root, so that no contact pair stores a row of the Jacobian over
    every dof.
    """
    Jx = qd.Vector.zero(gs.qd_float, 3)
    i_l_ = i_l
    while i_l_ > -1:
        I_l = [i_l_, i_b] if qd.static(sap_config.batch_links_info) else i_l_
        t_pos = pos - links_state.root_COM[i_l_, i_b]
        for i_d in range(links_info.dof_start[I_l], links_info.dof_end[I_l]):
            _, J_d = gu.qd_transform_motion_by_trans_quat(
                dofs_state.cdof_ang[i_d, i_b], dofs_state.cdof_vel[i_d, i_b], t_pos, gu.qd_identity_quat()
            )
            Jx = Jx + J_d * x[i_d, i_b]
        i_l_ = links_info.parent_idx[I_l]
    return Jx


@qd.func
def func_link_point_delassus(
    i_l0: int,
    i_l1: int,
    i_b: int,
    pos: qd.types.vector(3),
    dofs_state: DofsState,
    links_state: LinksState,
    links_info: LinksInfo,
    rigid_info: RigidInfo,
    sap_config: qd.template(),
):
    """
    Return the Delassus operator J M^-1 J^T in the world frame of a point attached to two links.

    J is the Jacobian of the velocity of the point `pos` attached to the link i_l0 minus the one of the same point
    attached to the link i_l1 (-1 for none). With the factorization M = L^T D L of the mass matrix, J M^-1 J^T = w^T
    D^-1 w where w = L^-T J^T. The rows of L only couple a dof to its ancestors, so that w is nonzero at the dofs of the
    two kinematic chains alone. The func walks each chain from its link to the root, the dofs in decreasing order,
    computing w0 and w1 of each chain apart and their cross terms at the dofs both chains share.
    """
    N_CHAIN = qd.static(sap_config.n_max_chain_dofs)
    chains_dof = qd.Vector([-1] * (2 * N_CHAIN), dt=gs.qd_int)
    chains_w = qd.Matrix.zero(gs.qd_float, 2 * N_CHAIN, 3)
    W = qd.Matrix.zero(gs.qd_float, 3, 3)
    for i_c in range(2):
        i_l_ = i_l0
        if i_c == 1:
            i_l_ = i_l1
        n_dofs = 0
        while i_l_ > -1:
            I_l = [i_l_, i_b] if qd.static(sap_config.batch_links_info) else i_l_
            t_pos = pos - links_state.root_COM[i_l_, i_b]
            for i_d_ in range(links_info.n_dofs[I_l]):
                i_d = links_info.dof_end[I_l] - 1 - i_d_
                _, w = gu.qd_transform_motion_by_trans_quat(
                    dofs_state.cdof_ang[i_d, i_b], dofs_state.cdof_vel[i_d, i_b], t_pos, gu.qd_identity_quat()
                )
                for i_k in range(n_dofs):
                    j_d = chains_dof[i_c * N_CHAIN + i_k]
                    L_ji = rigid_info.mass_mat_L[j_d, i_d, i_b]
                    for i in qd.static(range(3)):
                        w[i] = w[i] - L_ji * chains_w[i_c * N_CHAIN + i_k, i]
                chains_dof[i_c * N_CHAIN + n_dofs] = i_d
                for i in qd.static(range(3)):
                    chains_w[i_c * N_CHAIN + n_dofs, i] = w[i]
                n_dofs = n_dofs + 1
                W = W + rigid_info.mass_mat_D_inv[i_d, i_b] * w.outer_product(w)
            i_l_ = links_info.parent_idx[I_l]
    for i_k0 in range(N_CHAIN):
        i_d = chains_dof[i_k0]
        for i_k1 in range(N_CHAIN):
            if i_d >= 0 and i_d == chains_dof[N_CHAIN + i_k1]:
                w0 = qd.Vector([chains_w[i_k0, i] for i in qd.static(range(3))])
                w1 = qd.Vector([chains_w[N_CHAIN + i_k1, i] for i in qd.static(range(3))])
                W = W - rigid_info.mass_mat_D_inv[i_d, i_b] * (w0.outer_product(w1) + w1.outer_product(w0))
    return W


# ------------------------------------------------------------------------------------
# ----------------------------------- Newton solve -----------------------------------
# ------------------------------------------------------------------------------------
# The funcs of the solve take the same arguments after their own, the states, infos and configs of the solvers and of
# the coupler, then the constraint handlers.


@qd.kernel(graph=True)
def kernel_sap_solve(
    newton_counter: qd.types.ndarray(qd.i32, ndim=0),
    pcg_counter: qd.types.ndarray(qd.i32, ndim=0),
    linesearch_counter: qd.types.ndarray(qd.i32, ndim=0),
    fem_state: FEMState,
    dofs_state: DofsState,
    links_state: LinksState,
    sap_state: SAPState,
    fem_info: FEMInfo,
    vertex_constraints: FEMVertexConstraints,
    links_info: LinksInfo,
    rigid_info: RigidInfo,
    sap_info: SAPInfo,
    fem_config: qd.template(),
    rigid_config: qd.template(),
    sap_config: qd.template(),
    constraint_handlers: qd.template(),
    constraint_handler: qd.template(),
):
    """
    Run the Newton iterations of the SAP solve, each solving its step by preconditioned conjugate gradient and searching
    the step size along it by an exact line search.

    Each loop stops once no environment iterates on, or after the iterations its counter starts from. Each pass takes
    the iterations left past the current one into its `n_*_next` member (see SAPSolveLoops), and puts them back into its
    counter as soon as an environment iterates on. The loops run on the device as a graph, without a round-trip to the
    host per iteration.
    """
    # FIXME: quadrants#1083 - the statements before a nested graph do_while run again on every pass when the loop body
    # opens with bare statements, so each loop body opens with a for loop, at the cost of one launch per iteration.
    while qd.graph.do_while(newton_counter):
        for _ in range(1):
            sap_state.loops.n_newton_next[()] = newton_counter[()] - 1
            newton_counter[()] = 0
        # The gradient of the cost, then the convergence check
        func_init_unconstrained_gradient(
            fem_state,
            dofs_state,
            links_state,
            sap_state,
            fem_info,
            vertex_constraints,
            links_info,
            rigid_info,
            sap_info,
            fem_config,
            rigid_config,
            sap_config,
            constraint_handlers,
            constraint_handler,
        )
        func_compute_constraints(
            sap_state.envs.is_active,
            fem_state,
            dofs_state,
            links_state,
            sap_state,
            fem_info,
            vertex_constraints,
            links_info,
            rigid_info,
            sap_info,
            fem_config,
            rigid_config,
            sap_config,
            constraint_handlers,
            constraint_handler,
            op=ConstraintOp.GRADIENT_HESSIAN,
        )
        func_gather_constraints(
            sap_state.envs.is_active,
            fem_state,
            dofs_state,
            links_state,
            sap_state,
            fem_info,
            vertex_constraints,
            links_info,
            rigid_info,
            sap_info,
            fem_config,
            rigid_config,
            sap_config,
            constraint_handlers,
            constraint_handler,
            op=VertexDofOp.GRADIENT,
        )
        func_compute_vertices_dofs(
            sap_state.envs.is_active,
            fem_state,
            dofs_state,
            links_state,
            sap_state,
            fem_info,
            vertex_constraints,
            links_info,
            rigid_info,
            sap_info,
            fem_config,
            rigid_config,
            sap_config,
            constraint_handlers,
            constraint_handler,
            op=VertexDofOp.GRADIENT,
        )
        func_update_envs_active(newton_counter, sap_state, fem_info, sap_info, sap_config, constraint_handlers)

        # The Newton step
        func_init_pcg_solve(
            fem_state,
            dofs_state,
            links_state,
            sap_state,
            fem_info,
            vertex_constraints,
            links_info,
            rigid_info,
            sap_info,
            fem_config,
            rigid_config,
            sap_config,
            constraint_handlers,
            constraint_handler,
        )
        for _ in range(1):
            pcg_counter[()] = sap_info.n_pcg_iterations[None]
        while qd.graph.do_while(pcg_counter):
            for _ in range(1):
                sap_state.loops.n_pcg_next[()] = pcg_counter[()] - 1
                pcg_counter[()] = 0
            func_one_pcg_iter(
                pcg_counter,
                fem_state,
                dofs_state,
                links_state,
                sap_state,
                fem_info,
                vertex_constraints,
                links_info,
                rigid_info,
                sap_info,
                fem_config,
                rigid_config,
                sap_config,
                constraint_handlers,
                constraint_handler,
            )

        # The step size
        func_init_exact_linesearch(
            fem_state,
            dofs_state,
            links_state,
            sap_state,
            fem_info,
            vertex_constraints,
            links_info,
            rigid_info,
            sap_info,
            fem_config,
            rigid_config,
            sap_config,
            constraint_handlers,
            constraint_handler,
        )
        for _ in range(1):
            linesearch_counter[()] = sap_info.n_linesearch_iterations[None] + 1
        while qd.graph.do_while(linesearch_counter):
            for _ in range(1):
                sap_state.loops.n_linesearch_next[()] = linesearch_counter[()] - 1
                linesearch_counter[()] = 0
            func_one_exact_linesearch_iter(
                linesearch_counter,
                fem_state,
                dofs_state,
                links_state,
                sap_state,
                fem_info,
                vertex_constraints,
                links_info,
                rigid_info,
                sap_info,
                fem_config,
                rigid_config,
                sap_config,
                constraint_handlers,
                constraint_handler,
            )


@qd.func
def func_init_unconstrained_gradient(
    fem_state: FEMState,
    dofs_state: DofsState,
    links_state: LinksState,
    sap_state: SAPState,
    fem_info: FEMInfo,
    vertex_constraints: FEMVertexConstraints,
    links_info: LinksInfo,
    rigid_info: RigidInfo,
    sap_info: SAPInfo,
    fem_config: qd.template(),
    rigid_config: qd.template(),
    sap_config: qd.template(),
    constraint_handlers: qd.template(),
    constraint_handler: qd.template(),
):
    """
    Set the diagonal blocks of the Hessian of the FEM vertices to their free motion ones, and compute the part of the
    gradient the reference velocities set, A (v - v_ref) (see kernel_init_sap_solve).
    """
    if qd.static(sap_config.has_fem):
        for i_v, i_b in qd.ndrange(*sap_state.fem_vertices.v.shape, axes=qd.static(eoa(sap_state.fem_vertices.v))):
            # The FEM solver solves for positions and SAP for velocities, so the Hessian scales by dt^2
            sap_state.fem_vertices.diag3x3[i_v, i_b] = fem_state.vertices_solve.diag3x3[i_v, i_b] * fem_info.dt2[None]
            sap_state.fem_vertices.v_diff[i_v, i_b] = (
                sap_state.fem_vertices.v[i_v, i_b] - fem_state.vertices.vel_next[i_v, i_b]
            )
        func_compute_fem_matrix_vector_product(
            sap_state.fem_vertices.v_diff,
            sap_state.fem_vertices.gradient,
            sap_state.envs.is_active,
            fem_state,
            sap_state,
            fem_info,
            vertex_constraints,
            sap_info,
            fem_config,
        )
    if qd.static(sap_config.has_rigid):
        for i_d, i_b in qd.ndrange(
            sap_info.n_dofs[None], sap_state.envs.is_active.shape[0], axes=qd.static(eoa(sap_state.rigid_dofs.v))
        ):
            sap_state.rigid_dofs.v_diff[i_d, i_b] = sap_state.rigid_dofs.v[i_d, i_b] - dofs_state.vel[i_d, i_b]
        func_compute_rigid_mass_mat_vec_product(
            sap_state.rigid_dofs.v_diff, sap_state.rigid_dofs.gradient, sap_state.envs.is_active, rigid_info, sap_info
        )


@qd.func
def func_compute_vertices_dofs(
    envs_is_active: qd.Tensor,
    fem_state: FEMState,
    dofs_state: DofsState,
    links_state: LinksState,
    sap_state: SAPState,
    fem_info: FEMInfo,
    vertex_constraints: FEMVertexConstraints,
    links_info: LinksInfo,
    rigid_info: RigidInfo,
    sap_info: SAPInfo,
    fem_config: qd.template(),
    rigid_config: qd.template(),
    sap_config: qd.template(),
    constraint_handlers: qd.template(),
    constraint_handler: qd.template(),
    op: qd.template(),
):
    """
    Run an operation of the solve on the FEM vertices and the rigid dofs of the active environments, in one loop.

    `envs_is_active` holds the flags of the environments it runs on, and `op` is the operation, see VertexDofOp. Each
    chunk of the vertices of an entity or of the dofs (see CHUNK_SIZE) sums the values of its items into its partials,
    in a fixed order. The step size of the conjugate gradient update reads p^T A p from the partials of the previous
    operation, which every lane of a chunk adds up alike (see func_chunks_total).
    """
    # The sums of a chunk of vertices or dofs (see N_VERTEX_DOF_SUMS)
    N_SUMS = qd.static(8)
    # The items of a chunk (see CHUNK_SIZE)
    CHUNK_DIM = qd.static(32)
    N_LANES = qd.static(32 if sap_config.enable_cooperative_kernels else 1)
    SUMS_IDX = qd.static(VertexDofSum.of_op(op))
    for i_t in range(sap_state.vertices_dofs_partials.shape[2] * sap_state.vertices_dofs_partials.shape[0] * N_LANES):
        n_vertex_chunks = sap_info.n_vertex_chunks[None]
        n_chunks = sap_state.vertices_dofs_partials.shape[0]
        i_lane = i_t % N_LANES
        i_k = i_t // N_LANES % n_chunks
        i_b = i_t // (N_LANES * n_chunks)
        alpha = gs.qd_float(0.0)
        if qd.static(op == VertexDofOp.PCG_UPDATE):
            pTAp = func_chunks_total(
                i_b,
                0,
                sap_state.vertices_dofs_partials.shape[0],
                qd.static(VertexDofSum.PTAP),
                i_lane,
                sap_state,
                n_lanes=N_LANES,
                n_sums=1,
            )[0]
            # A direction along which the cost has no curvature, p^T A p underflowing, takes no step
            if envs_is_active[i_b] and pTAp > 0.0:
                alpha = sap_state.envs.rTz[i_b] / pTAp
        sums = qd.Vector.zero(gs.qd_float, N_SUMS)
        if envs_is_active[i_b]:
            for i_i_ in range(CHUNK_DIM // N_LANES):
                i_i = i_i_ * N_LANES + i_lane
                if qd.static(sap_config.has_fem):
                    if i_k < n_vertex_chunks:
                        i_v = sap_info.chunks_vertex_start[i_k] + i_i
                        if i_v < sap_info.chunks_vertex_end[i_k]:
                            sums = sums + func_compute_fem_vertex(
                                i_v,
                                i_b,
                                alpha,
                                fem_state,
                                sap_state,
                                fem_info,
                                vertex_constraints,
                                sap_info,
                                sap_config,
                                op=op,
                            )
                if qd.static(sap_config.has_rigid):
                    if i_k >= n_vertex_chunks:
                        i_d = (i_k - n_vertex_chunks) * CHUNK_DIM + i_i
                        if i_d < sap_info.n_dofs[None]:
                            sums = sums + func_compute_rigid_dof(
                                i_d,
                                i_b,
                                alpha,
                                dofs_state,
                                sap_state,
                                rigid_info,
                                sap_info,
                                rigid_config,
                                sap_config,
                                constraint_handler,
                                op=op,
                            )
        for i_q in qd.static(SUMS_IDX):
            if qd.static(N_LANES > 1):
                sums[i_q] = su.qd_block_sum(sums[i_q])
            if i_lane == 0 and envs_is_active[i_b]:
                sap_state.vertices_dofs_partials[i_k, i_q, i_b] = sums[i_q]


@qd.func
def func_compute_constraints(
    envs_is_active: qd.Tensor,
    fem_state: FEMState,
    dofs_state: DofsState,
    links_state: LinksState,
    sap_state: SAPState,
    fem_info: FEMInfo,
    vertex_constraints: FEMVertexConstraints,
    links_info: LinksInfo,
    rigid_info: RigidInfo,
    sap_info: SAPInfo,
    fem_config: qd.template(),
    rigid_config: qd.template(),
    sap_config: qd.template(),
    constraint_handlers: qd.template(),
    constraint_handler: qd.template(),
    op: qd.template(),
):
    """
    Run an operation of the solve on the constraints of every handler whose environment is active.

    `envs_is_active` holds the flags of the environments it runs on, and `op` is the operation, see ConstraintOp. The
    constraints of each environment run in their sorted order (see kernel_order_constraints) by chunks, each summing the
    values of its constraints into its partials in a fixed order, and each handler running its own branch, so that the
    operation takes one launch whatever the number of handlers. The matrix-vector product also computes the elastic
    corner forces along the search direction (see func_compute_fem_corner_force), independent work that shares the
    launch, its threads coming first. On GPU, the capacity of the chunks bounds the loop, since a bound read on the
    device launches a full grid at every call, while on CPU the threads take the chunks in use alone.
    """
    # The sums of a chunk of constraints (see N_CONSTRAINT_SUMS)
    N_SUMS = qd.static(3)
    # The items of a chunk (see CHUNK_SIZE)
    CHUNK_DIM = qd.static(32)
    N_LANES = qd.static(32 if sap_config.enable_cooperative_kernels else 1)
    HAS_ELEMENTS = qd.static(op == ConstraintOp.MATRIX_VECTOR_PRODUCT and sap_config.has_fem)
    n_envs = envs_is_active.shape[0]
    # The element threads fill whole subgroups, so that the lanes of a chunk sum their shares together
    n_elements = 0
    n_element_threads = 0
    if qd.static(HAS_ELEMENTS):
        n_elements = fem_info.elements.V.shape[0]
        n_element_threads = (n_envs * n_elements + CHUNK_DIM - 1) // CHUNK_DIM * CHUNK_DIM
    n_units = sap_state.constraint_order.units_env.shape[0]
    if qd.static(gs.backend == gs.cpu):
        n_units = sap_state.constraint_order.n_units[None]
    for i_t in range(n_element_threads + n_units * N_LANES):
        if qd.static(HAS_ELEMENTS):
            if i_t < n_envs * n_elements:
                i_b_e = i_t // n_elements
                if envs_is_active[i_b_e]:
                    func_compute_fem_corner_force(
                        i_t % n_elements,
                        i_b_e,
                        sap_state.fem_vertices.p,
                        fem_state,
                        sap_state,
                        fem_info,
                        vertex_constraints,
                        sap_info,
                        fem_config,
                    )
        i_lane = i_t % N_LANES
        i_u = qd.max(i_t - n_element_threads, 0) // N_LANES
        i_b = sap_state.constraint_order.units_env[i_u]
        is_active = i_t >= n_element_threads and i_u < sap_state.constraint_order.n_units[None] and envs_is_active[i_b]
        sums = qd.Vector.zero(gs.qd_float, N_SUMS)
        if is_active:
            i_s_end = sap_state.constraint_order.envs_constraint_start[i_b + 1]
            for i_i_ in range(CHUNK_DIM // N_LANES):
                i_s = sap_state.constraint_order.units_start[i_u] + i_i_ * N_LANES + i_lane
                if i_s < i_s_end:
                    i_q = sap_state.constraint_order.order[i_s]
                    for i_h, handler in qd.static(enumerate(constraint_handlers)):
                        if sap_info.handlers_slot_start[i_h] <= i_q and i_q < sap_info.handlers_slot_start[i_h + 1]:
                            i_c = i_q - sap_info.handlers_slot_start[i_h]
                            dell, d2ell, energy = handler.compute_solve_op(
                                i_c,
                                dofs_state,
                                links_state,
                                sap_state,
                                fem_info,
                                links_info,
                                sap_info,
                                sap_config,
                                op=op,
                            )
                            if qd.static(op == ConstraintOp.LINE_DERIVATIVES):
                                sums[ConstraintSum.DELL] = sums[ConstraintSum.DELL] + dell
                                sums[ConstraintSum.D2ELL] = sums[ConstraintSum.D2ELL] + d2ell
                            if qd.static(op == ConstraintOp.ENERGY):
                                sums[ConstraintSum.ENERGY] = sums[ConstraintSum.ENERGY] + energy
        for i_q in qd.static(ConstraintSum.of_op(op)):
            if qd.static(N_LANES > 1):
                sums[i_q] = su.qd_block_sum(sums[i_q])
            if i_lane == 0 and is_active:
                sap_state.constraints_partials[i_u, i_q] = sums[i_q]


@qd.func
def func_gather_constraints(
    envs_is_active: qd.Tensor,
    fem_state: FEMState,
    dofs_state: DofsState,
    links_state: LinksState,
    sap_state: SAPState,
    fem_info: FEMInfo,
    vertex_constraints: FEMVertexConstraints,
    links_info: LinksInfo,
    rigid_info: RigidInfo,
    sap_info: SAPInfo,
    fem_config: qd.template(),
    rigid_config: qd.template(),
    sap_config: qd.template(),
    constraint_handlers: qd.template(),
    constraint_handler: qd.template(),
    op: qd.template(),
):
    """
    Gather the contacts into the wrench of each rigid link and into each FEM vertex, in one loop.

    The threads of the links come first, then those of the vertices (see func_compute_link_wrench and
    func_gather_fem_vertex), independent work that shares one launch. `op` is the operation, see VertexDofOp.
    """
    # The lanes gathering a vertex (see func_gather_fem_vertex)
    LOG2_GATHER_LANES = qd.static(3)
    HAS_LINKS = qd.static(sap_config.has_rigid_contacts and op != VertexDofOp.LINESEARCH_D2ELLA)
    # The wrenches hold a row per share of each link, which the lanes of a chunk take together
    N_LINK_LANES = qd.static(32 if sap_config.enable_cooperative_kernels else 1)
    N_VERTEX_LANES = qd.static(2**LOG2_GATHER_LANES if sap_config.enable_cooperative_kernels else 1)
    n_envs = envs_is_active.shape[0]
    n_link_threads = 0
    if qd.static(HAS_LINKS):
        n_link_threads = sap_state.constraint_order.links_wrench_force.shape[0] * n_envs * N_LINK_LANES
    n_vertex_threads = 0
    if qd.static(sap_config.has_fem):
        n_vertex_threads = n_envs * sap_state.fem_vertices.v.shape[0] * N_VERTEX_LANES
    for i_t in range(n_link_threads + n_vertex_threads):
        if qd.static(HAS_LINKS):
            if i_t < n_link_threads:
                func_compute_link_wrench(
                    i_t, envs_is_active, links_state, sap_state, sap_info, sap_config, constraint_handlers, op=op
                )
        if qd.static(sap_config.has_fem):
            if i_t >= n_link_threads:
                func_gather_fem_vertex(
                    i_t - n_link_threads,
                    envs_is_active,
                    sap_state,
                    fem_info,
                    vertex_constraints,
                    sap_info,
                    sap_config,
                    constraint_handlers,
                    op=op,
                )


@qd.func
def func_compute_fem_vertex(
    i_v: int,
    i_b: int,
    alpha: float,
    fem_state: FEMState,
    sap_state: SAPState,
    fem_info: FEMInfo,
    vertex_constraints: FEMVertexConstraints,
    sap_info: SAPInfo,
    sap_config: qd.template(),
    op: qd.template(),
):
    """
    Run an operation of the solve on a FEM vertex, and return its share of the sums of the operation.

    The vertex is i_v of the environment i_b, and the sums those of func_compute_vertices_dofs (see VertexDofSum). The
    preconditioned residual z adds the rigid motion of the entity of each free vertex solving the residual restricted to
    the rigid motions (see SAPFEMCoarse), which the per-environment loops solve between the loops computing r and those
    using z. `alpha` is the step size of the conjugate gradient update.
    """
    # The sums of a chunk of vertices or dofs (see N_VERTEX_DOF_SUMS)
    N_SUMS = qd.static(8)
    sums = qd.Vector.zero(gs.qd_float, N_SUMS)
    i_e = sap_info.fem_vertices_entity_idx[i_v]
    is_free = func_is_fem_vertex_free(i_v, i_b, vertex_constraints, sap_config)
    d = fem_state.vertices.pos[i_v, i_b] - sap_state.fem_coarse.x_ref[i_e, i_b]
    dt2 = fem_info.dt2[None]
    inertia = fem_info.vertices.mass_over_dt2[i_v] * dt2 * (fem_info.damping_alpha_dt[None] + 1.0)
    if qd.static(op == VertexDofOp.GRADIENT):
        # The norms weigh each vertex by the diagonal block D of the Hessian of its free motion, so that the residual
        # left on a stiff entity counts by the velocity error it leaves.
        D = fem_state.vertices_solve.diag3x3[i_v, i_b] * dt2
        D_inv = fem_state.vertices_solve.prec[i_v, i_b] / dt2
        gradient = sap_state.fem_vertices.gradient[i_v, i_b]
        impulse = sap_state.fem_vertices.impulse[i_v, i_b]
        v = sap_state.fem_vertices.v[i_v, i_b]
        sums[VertexDofSum.GRADIENT_NORM] = gradient.dot(D_inv @ gradient)
        sums[VertexDofSum.MOMENTUM_NORM] = v.dot(D @ v)
        sums[VertexDofSum.IMPULSE_NORM] = impulse.dot(D_inv @ impulse)
    if qd.static(op == VertexDofOp.PCG_INIT):
        r = -sap_state.fem_vertices.gradient[i_v, i_b]
        z = sap_state.fem_vertices.prec[i_v, i_b] @ r
        sap_state.fem_vertices.x[i_v, i_b] = qd.Vector.zero(gs.qd_float, 3)
        sap_state.fem_vertices.r[i_v, i_b] = r
        sap_state.fem_vertices.z[i_v, i_b] = z
        sap_state.fem_vertices.p[i_v, i_b] = qd.Vector.zero(gs.qd_float, 3)
        sums[VertexDofSum.RTZ] = r.dot(z)
    if qd.static(op == VertexDofOp.PCG_SEARCH_DIRECTION):
        z = sap_state.fem_vertices.z[i_v, i_b]
        if is_free:
            sol = sap_state.fem_coarse.sol[i_e, i_b]
            c_lin = qd.Vector([sol[0], sol[1], sol[2]])
            c_ang = qd.Vector([sol[3], sol[4], sol[5]])
            z = z + c_lin + c_ang.cross(d)
        p = z + sap_state.envs.beta[i_b] * sap_state.fem_vertices.p[i_v, i_b]
        sap_state.fem_vertices.p[i_v, i_b] = p
        sap_state.fem_vertices.Ap[i_v, i_b] = inertia * p
    if qd.static(op == VertexDofOp.PCG_PTAP):
        sums[VertexDofSum.PTAP] = sap_state.fem_vertices.p[i_v, i_b].dot(sap_state.fem_vertices.Ap[i_v, i_b])
    if qd.static(op == VertexDofOp.PCG_UPDATE):
        r = sap_state.fem_vertices.r[i_v, i_b] - alpha * sap_state.fem_vertices.Ap[i_v, i_b]
        z = sap_state.fem_vertices.prec[i_v, i_b] @ r
        sap_state.fem_vertices.x[i_v, i_b] = (
            sap_state.fem_vertices.x[i_v, i_b] + alpha * sap_state.fem_vertices.p[i_v, i_b]
        )
        sap_state.fem_vertices.r[i_v, i_b] = r
        sap_state.fem_vertices.z[i_v, i_b] = z
        sums[VertexDofSum.RTZ] = r.dot(z)
    if qd.static(op == VertexDofOp.PCG_INIT or op == VertexDofOp.PCG_UPDATE):
        # The residual restricted to the rigid motions, P^T r = (sum r, sum d x r) over the free vertices
        if is_free:
            r = sap_state.fem_vertices.r[i_v, i_b]
            dxr = d.cross(r)
            for k in qd.static(range(3)):
                sums[VertexDofSum.COARSE_RHS + k] = r[k]
                sums[VertexDofSum.COARSE_RHS + 3 + k] = dxr[k]
    if qd.static(op == VertexDofOp.LINESEARCH_INIT):
        sums[VertexDofSum.SLOPE] = sap_state.fem_vertices.x[i_v, i_b].dot(sap_state.fem_vertices.gradient[i_v, i_b])
        sap_state.fem_vertices.x_prev[i_v, i_b] = sap_state.fem_vertices.v[i_v, i_b]
    if qd.static(op == VertexDofOp.ENERGY):
        v_diff = sap_state.fem_vertices.v[i_v, i_b] - fem_state.vertices.vel_next[i_v, i_b]
        sap_state.fem_vertices.v_diff[i_v, i_b] = v_diff
        sums[VertexDofSum.ENERGY] = 0.5 * inertia * v_diff.norm_sqr()
    if qd.static(op == VertexDofOp.LINESEARCH_D2ELLA):
        sums[VertexDofSum.D2ELLA] = sap_state.fem_vertices.x[i_v, i_b].dot(sap_state.fem_vertices.dp[i_v, i_b])
    if qd.static(op == VertexDofOp.LINESEARCH_VELOCITY):
        v = sap_state.fem_vertices.x_prev[i_v, i_b] + sap_state.envs.step_size[i_b] * sap_state.fem_vertices.x[i_v, i_b]
        sap_state.fem_vertices.v[i_v, i_b] = v
        v_ref = fem_state.vertices.vel_next[i_v, i_b]
        sums[VertexDofSum.DELL] = sap_state.fem_vertices.dp[i_v, i_b].dot(v - v_ref)
    return sums


@qd.func
def func_compute_rigid_dof(
    i_d: int,
    i_b: int,
    alpha: float,
    dofs_state: DofsState,
    sap_state: SAPState,
    rigid_info: RigidInfo,
    sap_info: SAPInfo,
    rigid_config: qd.template(),
    sap_config: qd.template(),
    constraint_handler: qd.template(),
    op: qd.template(),
):
    """
    Run an operation of the solve on a rigid dof, and return its share of the sums of the operation.

    The dof is i_d of the environment i_b, and the sums those of func_compute_vertices_dofs (see VertexDofSum). The
    factor of the mass matrix the rigid solver holds preconditions the rigid dofs, its solve running on the dofs of a
    mass block in a row from its first dof, the others leaving it to that dof. That factor holds the implicit damping of
    some integrators, which the Hessian of the solve leaves out, so that the Hessian applies the mass matrix itself, as
    M p = M z + beta M p_prev: the product reads z, which no dof writes there, so that each dof updates its own share of
    M p without the search direction of the others.
    """
    # The sums of a chunk of vertices or dofs (see N_VERTEX_DOF_SUMS)
    N_SUMS = qd.static(8)
    sums = qd.Vector.zero(gs.qd_float, N_SUMS)
    if qd.static(op == VertexDofOp.GRADIENT):
        Jt_gamma = func_gather_rigid_dof(
            i_d, i_b, dofs_state, sap_state, sap_info, sap_config, constraint_handler, op=op
        )
        gradient = sap_state.rigid_dofs.gradient[i_d, i_b] - Jt_gamma
        sap_state.rigid_dofs.gradient[i_d, i_b] = gradient
        sap_state.rigid_dofs.impulse[i_d, i_b] = Jt_gamma
        mass = rigid_info.mass_mat[i_d, i_d, i_b]
        sums[VertexDofSum.GRADIENT_NORM] = gradient**2 / mass
        sums[VertexDofSum.MOMENTUM_NORM] = sap_state.rigid_dofs.v[i_d, i_b] ** 2 * mass
        sums[VertexDofSum.IMPULSE_NORM] = Jt_gamma**2 / mass
    if qd.static(op == VertexDofOp.PCG_INIT or op == VertexDofOp.PCG_UPDATE):
        if rigid_info.dofs_mass_block_start[i_d] == i_d:
            block_dof_end = rigid_info.dofs_mass_block_end[i_d]
            for j_d in range(i_d, block_dof_end):
                if qd.static(op == VertexDofOp.PCG_INIT):
                    sap_state.rigid_dofs.r[j_d, i_b] = -sap_state.rigid_dofs.gradient[j_d, i_b]
                    sap_state.rigid_dofs.x[j_d, i_b] = 0.0
                    sap_state.rigid_dofs.p[j_d, i_b] = 0.0
                    sap_state.rigid_dofs.Mp[j_d, i_b] = 0.0
                else:
                    sap_state.rigid_dofs.r[j_d, i_b] = (
                        sap_state.rigid_dofs.r[j_d, i_b] - alpha * sap_state.rigid_dofs.Ap[j_d, i_b]
                    )
                    sap_state.rigid_dofs.x[j_d, i_b] = (
                        sap_state.rigid_dofs.x[j_d, i_b] + alpha * sap_state.rigid_dofs.p[j_d, i_b]
                    )
            func_solve_mass_block(
                i_b, i_d, block_dof_end, sap_state.rigid_dofs.r, sap_state.rigid_dofs.z, rigid_info, rigid_config
            )
            rTz = gs.qd_float(0.0)
            for j_d in range(i_d, block_dof_end):
                rTz = rTz + sap_state.rigid_dofs.r[j_d, i_b] * sap_state.rigid_dofs.z[j_d, i_b]
            sums[VertexDofSum.RTZ] = rTz
    if qd.static(op == VertexDofOp.PCG_SEARCH_DIRECTION):
        beta = sap_state.envs.beta[i_b]
        Mz = gs.qd_float(0.0)
        for j_d in range(rigid_info.dofs_mass_block_start[i_d], rigid_info.dofs_mass_block_end[i_d]):
            Mz = Mz + rigid_info.mass_mat[i_d, j_d, i_b] * sap_state.rigid_dofs.z[j_d, i_b]
        sap_state.rigid_dofs.p[i_d, i_b] = sap_state.rigid_dofs.z[i_d, i_b] + beta * sap_state.rigid_dofs.p[i_d, i_b]
        sap_state.rigid_dofs.Mp[i_d, i_b] = Mz + beta * sap_state.rigid_dofs.Mp[i_d, i_b]
    if qd.static(op == VertexDofOp.PCG_PTAP):
        Ap = sap_state.rigid_dofs.Mp[i_d, i_b] + func_gather_rigid_dof(
            i_d, i_b, dofs_state, sap_state, sap_info, sap_config, constraint_handler, op=op
        )
        sap_state.rigid_dofs.Ap[i_d, i_b] = Ap
        sums[VertexDofSum.PTAP] = sap_state.rigid_dofs.p[i_d, i_b] * Ap
    if qd.static(op == VertexDofOp.LINESEARCH_INIT):
        sums[VertexDofSum.SLOPE] = sap_state.rigid_dofs.x[i_d, i_b] * sap_state.rigid_dofs.gradient[i_d, i_b]
        sap_state.rigid_dofs.x_prev[i_d, i_b] = sap_state.rigid_dofs.v[i_d, i_b]
    if qd.static(op == VertexDofOp.ENERGY):
        sums[VertexDofSum.ENERGY] = (
            0.5 * sap_state.rigid_dofs.v_diff[i_d, i_b] * sap_state.rigid_dofs.mass_v_diff[i_d, i_b]
        )
    if qd.static(op == VertexDofOp.LINESEARCH_D2ELLA):
        sums[VertexDofSum.D2ELLA] = sap_state.rigid_dofs.x[i_d, i_b] * sap_state.rigid_dofs.dp[i_d, i_b]
    if qd.static(op == VertexDofOp.LINESEARCH_VELOCITY):
        v = sap_state.rigid_dofs.x_prev[i_d, i_b] + sap_state.envs.step_size[i_b] * sap_state.rigid_dofs.x[i_d, i_b]
        sap_state.rigid_dofs.v[i_d, i_b] = v
        sums[VertexDofSum.DELL] = sap_state.rigid_dofs.dp[i_d, i_b] * (v - dofs_state.vel[i_d, i_b])
    return sums


@qd.func
def func_init_pcg_solve(
    fem_state: FEMState,
    dofs_state: DofsState,
    links_state: LinksState,
    sap_state: SAPState,
    fem_info: FEMInfo,
    vertex_constraints: FEMVertexConstraints,
    links_info: LinksInfo,
    rigid_info: RigidInfo,
    sap_info: SAPInfo,
    fem_config: qd.template(),
    rigid_config: qd.template(),
    sap_config: qd.template(),
    constraint_handlers: qd.template(),
    constraint_handler: qd.template(),
):
    """
    Start the conjugate gradient of the active environments from the sums of their residuals.

    The lanes of a chunk (see CHUNK_SIZE) take an environment. The Newton step only needs to be accurate relative to the
    gradient it reduces. An absolute bound on the residual stops the solve before it moves the light vertices, whose
    residual impulses are small while their velocity errors are large, and the Newton iterations stall above their
    convergence threshold. The FEM vertices and the rigid dofs reduce their own residual each, since the exact
    preconditioner of the rigid dofs solves their share of r^T z within an iteration, which would stop a solve on the
    sum early.
    """
    N_LANES = qd.static(32 if sap_config.enable_cooperative_kernels else 1)
    for i_b in range(sap_state.envs.is_active.shape[0]):
        sap_state.envs.is_pcg_active[i_b] = sap_state.envs.is_active[i_b]
        # The first search direction is the preconditioned residual, which the first iteration sets from p = 0
        sap_state.envs.beta[i_b] = 0.0
    func_compute_vertices_dofs(
        sap_state.envs.is_pcg_active,
        fem_state,
        dofs_state,
        links_state,
        sap_state,
        fem_info,
        vertex_constraints,
        links_info,
        rigid_info,
        sap_info,
        fem_config,
        rigid_config,
        sap_config,
        constraint_handlers,
        constraint_handler,
        op=VertexDofOp.PCG_INIT,
    )
    for i_t in range(sap_state.envs.is_active.shape[0] * N_LANES):
        i_lane = i_t % N_LANES
        i_b = i_t // N_LANES
        is_active = sap_state.envs.is_pcg_active[i_b]
        rTz_fem, rTz_rigid = func_pcg_residuals(i_b, i_lane, is_active, sap_state, sap_info, sap_config)
        if is_active and i_lane == 0:
            sap_state.envs.rTz[i_b] = rTz_fem + rTz_rigid
            sap_state.envs.rTz_fem_threshold[i_b] = sap_info.pcg_threshold[None] * rTz_fem
            sap_state.envs.rTz_rigid_threshold[i_b] = sap_info.pcg_threshold[None] * rTz_rigid
            sap_state.envs.is_pcg_active[i_b] = sap_state.envs.rTz[i_b] > 0.0


@qd.func
def func_pcg_residuals(
    i_b: int, i_lane: int, is_active: bool, sap_state: SAPState, sap_info: SAPInfo, sap_config: qd.template()
):
    """
    Return the products r^T z of the FEM vertices and of the rigid dofs of an environment.

    The product of the vertices holds the share of the rigid motions of the FEM entities, whose solutions the first lane
    stores for an active environment. Every lane of a chunk calls it for the environment i_b, adding up the residuals as
    in func_chunks_total. The residual of each FEM entity restricted to its rigid motions is solved by the inverse of
    its coarse Hessian (see SAPFEMCoarse).
    """
    N_LANES = qd.static(32 if sap_config.enable_cooperative_kernels else 1)
    rTz_fem = func_chunks_total(
        i_b,
        0,
        sap_info.n_vertex_chunks[None],
        qd.static(VertexDofSum.RTZ),
        i_lane,
        sap_state,
        n_lanes=N_LANES,
        n_sums=1,
    )[0]
    if qd.static(sap_config.has_fem):
        for i_e in range(sap_state.fem_coarse.mass.shape[0]):
            rhs = func_chunks_total(
                i_b,
                sap_info.entities_chunk_start[i_e],
                sap_info.entities_chunk_start[i_e + 1],
                qd.static(VertexDofSum.COARSE_RHS),
                i_lane,
                sap_state,
                n_lanes=N_LANES,
                n_sums=6,
            )
            if is_active:
                sol = sap_state.fem_coarse.A_inv[i_e, i_b] @ rhs
                if i_lane == 0:
                    sap_state.fem_coarse.sol[i_e, i_b] = sol
                rTz_fem = rTz_fem + rhs.dot(sol)
    rTz_rigid = func_chunks_total(
        i_b,
        sap_info.n_vertex_chunks[None],
        sap_state.vertices_dofs_partials.shape[0],
        qd.static(VertexDofSum.RTZ),
        i_lane,
        sap_state,
        n_lanes=N_LANES,
        n_sums=1,
    )[0]
    return rTz_fem, rTz_rigid


@qd.func
def func_one_pcg_iter(
    pcg_counter: qd.Tensor,
    fem_state: FEMState,
    dofs_state: DofsState,
    links_state: LinksState,
    sap_state: SAPState,
    fem_info: FEMInfo,
    vertex_constraints: FEMVertexConstraints,
    links_info: LinksInfo,
    rigid_info: RigidInfo,
    sap_info: SAPInfo,
    fem_config: qd.template(),
    rigid_config: qd.template(),
    sap_config: qd.template(),
    constraint_handlers: qd.template(),
    constraint_handler: qd.template(),
):
    """
    Run one iteration of the preconditioned conjugate gradient solve of the Newton step.

    The search direction p of the iteration is updated in the loop computing the inertial part of A p, the vertices and
    the dofs gather the elastic and the constraint parts, then p^T A p, and the step size is computed in the loop
    stepping the solution, so that the iteration takes the fewest loops. The environments whose residual is small enough
    stop, the others taking beta.

    See https://en.wikipedia.org/wiki/Conjugate_gradient_method#The_preconditioned_conjugate_gradient_method
    """
    N_LANES = qd.static(32 if sap_config.enable_cooperative_kernels else 1)
    func_compute_vertices_dofs(
        sap_state.envs.is_pcg_active,
        fem_state,
        dofs_state,
        links_state,
        sap_state,
        fem_info,
        vertex_constraints,
        links_info,
        rigid_info,
        sap_info,
        fem_config,
        rigid_config,
        sap_config,
        constraint_handlers,
        constraint_handler,
        op=VertexDofOp.PCG_SEARCH_DIRECTION,
    )
    func_compute_constraints(
        sap_state.envs.is_pcg_active,
        fem_state,
        dofs_state,
        links_state,
        sap_state,
        fem_info,
        vertex_constraints,
        links_info,
        rigid_info,
        sap_info,
        fem_config,
        rigid_config,
        sap_config,
        constraint_handlers,
        constraint_handler,
        op=ConstraintOp.MATRIX_VECTOR_PRODUCT,
    )
    func_gather_constraints(
        sap_state.envs.is_pcg_active,
        fem_state,
        dofs_state,
        links_state,
        sap_state,
        fem_info,
        vertex_constraints,
        links_info,
        rigid_info,
        sap_info,
        fem_config,
        rigid_config,
        sap_config,
        constraint_handlers,
        constraint_handler,
        op=VertexDofOp.PCG_PTAP,
    )
    func_compute_vertices_dofs(
        sap_state.envs.is_pcg_active,
        fem_state,
        dofs_state,
        links_state,
        sap_state,
        fem_info,
        vertex_constraints,
        links_info,
        rigid_info,
        sap_info,
        fem_config,
        rigid_config,
        sap_config,
        constraint_handlers,
        constraint_handler,
        op=VertexDofOp.PCG_PTAP,
    )
    func_compute_vertices_dofs(
        sap_state.envs.is_pcg_active,
        fem_state,
        dofs_state,
        links_state,
        sap_state,
        fem_info,
        vertex_constraints,
        links_info,
        rigid_info,
        sap_info,
        fem_config,
        rigid_config,
        sap_config,
        constraint_handlers,
        constraint_handler,
        op=VertexDofOp.PCG_UPDATE,
    )
    for i_t in range(sap_state.envs.is_active.shape[0] * N_LANES):
        i_lane = i_t % N_LANES
        i_b = i_t // N_LANES
        is_active = sap_state.envs.is_pcg_active[i_b]
        rTz_fem, rTz_rigid = func_pcg_residuals(i_b, i_lane, is_active, sap_state, sap_info, sap_config)
        if is_active and i_lane == 0:
            is_active = (
                rTz_fem > sap_state.envs.rTz_fem_threshold[i_b] or rTz_rigid > sap_state.envs.rTz_rigid_threshold[i_b]
            )
            sap_state.envs.is_pcg_active[i_b] = is_active
            if is_active:
                pcg_counter[()] = sap_state.loops.n_pcg_next[()]
                sap_state.envs.beta[i_b] = (rTz_fem + rTz_rigid) / sap_state.envs.rTz[i_b]
                sap_state.envs.rTz[i_b] = rTz_fem + rTz_rigid


@qd.func
def func_init_exact_linesearch(
    fem_state: FEMState,
    dofs_state: DofsState,
    links_state: LinksState,
    sap_state: SAPState,
    fem_info: FEMInfo,
    vertex_constraints: FEMVertexConstraints,
    links_info: LinksInfo,
    rigid_info: RigidInfo,
    sap_info: SAPInfo,
    fem_config: qd.template(),
    rigid_config: qd.template(),
    sap_config: qd.template(),
    constraint_handlers: qd.template(),
    constraint_handler: qd.template(),
):
    """
    Start the exact line search of the active environments along the Newton step, from the largest step size.

    The slope, the cost and the second derivative of the free motion part of the cost along the step are computed at
    its start, for the first pass of the line search (see func_one_exact_linesearch_iter).
    """
    # The items of a chunk (see CHUNK_SIZE)
    CHUNK_DIM = qd.static(32)
    N_LANES = qd.static(32 if sap_config.enable_cooperative_kernels else 1)
    for i_b in range(sap_state.envs.is_active.shape[0]):
        sap_state.envs.is_linesearch_active[i_b] = sap_state.envs.is_active[i_b]
        if sap_state.envs.is_linesearch_active[i_b]:
            sap_state.envs.step_size[i_b] = sap_info.linesearch_max_step_size[None]
    func_compute_vertices_dofs(
        sap_state.envs.is_linesearch_active,
        fem_state,
        dofs_state,
        links_state,
        sap_state,
        fem_info,
        vertex_constraints,
        links_info,
        rigid_info,
        sap_info,
        fem_config,
        rigid_config,
        sap_config,
        constraint_handlers,
        constraint_handler,
        op=VertexDofOp.LINESEARCH_INIT,
    )
    # The shares of the cost of the solve at the start of the line search
    if qd.static(sap_config.has_rigid):
        for i_d, i_b in qd.ndrange(
            sap_info.n_dofs[None], sap_state.envs.is_active.shape[0], axes=qd.static(eoa(sap_state.rigid_dofs.v))
        ):
            if sap_state.envs.is_linesearch_active[i_b]:
                sap_state.rigid_dofs.v_diff[i_d, i_b] = sap_state.rigid_dofs.v[i_d, i_b] - dofs_state.vel[i_d, i_b]
        func_compute_rigid_mass_mat_vec_product(
            sap_state.rigid_dofs.v_diff,
            sap_state.rigid_dofs.mass_v_diff,
            sap_state.envs.is_linesearch_active,
            rigid_info,
            sap_info,
        )
    func_compute_vertices_dofs(
        sap_state.envs.is_linesearch_active,
        fem_state,
        dofs_state,
        links_state,
        sap_state,
        fem_info,
        vertex_constraints,
        links_info,
        rigid_info,
        sap_info,
        fem_config,
        rigid_config,
        sap_config,
        constraint_handlers,
        constraint_handler,
        op=VertexDofOp.ENERGY,
    )
    if qd.static(sap_config.has_fem):
        # The elastic energy of the FEM elements of each environment, summed by chunks in a fixed order
        for i_t in range(sap_state.elements_partials.shape[1] * sap_state.elements_partials.shape[0] * N_LANES):
            n_element_chunks = sap_state.elements_partials.shape[0]
            n_elements = fem_info.elements.V.shape[0]
            i_lane = i_t % N_LANES
            i_k = i_t // N_LANES % n_element_chunks
            i_b = i_t // (N_LANES * n_element_chunks)
            energy = gs.qd_float(0.0)
            if sap_state.envs.is_linesearch_active[i_b]:
                for i_i_ in range(CHUNK_DIM // N_LANES):
                    i_e = i_k * CHUNK_DIM + i_i_ * N_LANES + i_lane
                    if i_e < n_elements:
                        S = func_element_mapping_matrix(i_e, i_b, fem_info, vertex_constraints, fem_config)
                        i_vs = fem_info.elements.el2v[i_e]
                        x = qd.Matrix.cols([sap_state.fem_vertices.v_diff[i_vs[i], i_b] for i in qd.static(range(4))])
                        dF, dP = func_elastic_product(i_e, i_b, S, x, fem_state, fem_info, fem_config)
                        V_dt2 = fem_info.elements.V[i_e] * fem_info.dt2[None]
                        energy = energy + 0.5 * (dF * dP).sum() * fem_info.damping_beta_factor[None] * V_dt2
            if qd.static(N_LANES > 1):
                energy = su.qd_block_sum(energy)
            if i_lane == 0 and sap_state.envs.is_linesearch_active[i_b]:
                sap_state.elements_partials[i_k, i_b] = energy
    func_compute_constraints(
        sap_state.envs.is_linesearch_active,
        fem_state,
        dofs_state,
        links_state,
        sap_state,
        fem_info,
        vertex_constraints,
        links_info,
        rigid_info,
        sap_info,
        fem_config,
        rigid_config,
        sap_config,
        constraint_handlers,
        constraint_handler,
        op=ConstraintOp.ENERGY,
    )
    # The product of the Hessian of the free motion with the Newton step
    if qd.static(sap_config.has_fem):
        func_compute_fem_matrix_vector_product(
            sap_state.fem_vertices.x,
            sap_state.fem_vertices.dp,
            sap_state.envs.is_linesearch_active,
            fem_state,
            sap_state,
            fem_info,
            vertex_constraints,
            sap_info,
            fem_config,
        )
    if qd.static(sap_config.has_rigid):
        func_compute_rigid_mass_mat_vec_product(
            sap_state.rigid_dofs.x, sap_state.rigid_dofs.dp, sap_state.envs.is_linesearch_active, rigid_info, sap_info
        )
    func_compute_constraints(
        sap_state.envs.is_linesearch_active,
        fem_state,
        dofs_state,
        links_state,
        sap_state,
        fem_info,
        vertex_constraints,
        links_info,
        rigid_info,
        sap_info,
        fem_config,
        rigid_config,
        sap_config,
        constraint_handlers,
        constraint_handler,
        op=ConstraintOp.SEARCH_DIRECTION,
    )
    func_gather_constraints(
        sap_state.envs.is_linesearch_active,
        fem_state,
        dofs_state,
        links_state,
        sap_state,
        fem_info,
        vertex_constraints,
        links_info,
        rigid_info,
        sap_info,
        fem_config,
        rigid_config,
        sap_config,
        constraint_handlers,
        constraint_handler,
        op=VertexDofOp.LINESEARCH_D2ELLA,
    )
    func_compute_vertices_dofs(
        sap_state.envs.is_linesearch_active,
        fem_state,
        dofs_state,
        links_state,
        sap_state,
        fem_info,
        vertex_constraints,
        links_info,
        rigid_info,
        sap_info,
        fem_config,
        rigid_config,
        sap_config,
        constraint_handlers,
        constraint_handler,
        op=VertexDofOp.LINESEARCH_D2ELLA,
    )


@qd.func
def func_one_exact_linesearch_iter(
    linesearch_counter: qd.Tensor,
    fem_state: FEMState,
    dofs_state: DofsState,
    links_state: LinksState,
    sap_state: SAPState,
    fem_info: FEMInfo,
    vertex_constraints: FEMVertexConstraints,
    links_info: LinksInfo,
    rigid_info: RigidInfo,
    sap_info: SAPInfo,
    fem_config: qd.template(),
    rigid_config: qd.template(),
    sap_config: qd.template(),
    constraint_handlers: qd.template(),
    constraint_handler: qd.template(),
):
    """
    Run one pass of the exact line search using rtsafe, a Newton-Raphson iteration safeguarded by bisection.

    Each pass steps the velocities by the current step size and evaluates the derivative of the cost along the Newton
    step f and its own derivative df there. The first pass evaluates the largest step size. An environment whose Newton
    iteration decreased its cost by no more than its round-off stops its solve there, one whose cost still decreases at
    the largest step size takes it, and the others start the bracket [alpha_min, alpha_max] holding the root of f. The
    next passes shrink the bracket to the step size on the side where f has the sign it has at that end, and the next
    step size takes a Newton step on f, or bisects the bracket when the Newton step leaves it or reduces f too slowly.
    """
    func_compute_vertices_dofs(
        sap_state.envs.is_linesearch_active,
        fem_state,
        dofs_state,
        links_state,
        sap_state,
        fem_info,
        vertex_constraints,
        links_info,
        rigid_info,
        sap_info,
        fem_config,
        rigid_config,
        sap_config,
        constraint_handlers,
        constraint_handler,
        op=VertexDofOp.LINESEARCH_VELOCITY,
    )
    func_compute_constraints(
        sap_state.envs.is_linesearch_active,
        fem_state,
        dofs_state,
        links_state,
        sap_state,
        fem_info,
        vertex_constraints,
        links_info,
        rigid_info,
        sap_info,
        fem_config,
        rigid_config,
        sap_config,
        constraint_handlers,
        constraint_handler,
        op=ConstraintOp.LINE_DERIVATIVES,
    )
    for i_b in range(sap_state.envs.is_active.shape[0]):
        if not sap_state.envs.is_linesearch_active[i_b]:
            continue
        # The first pass evaluates the largest step size, which the kernel counts as an extra iteration
        is_first_pass = sap_state.loops.n_linesearch_next[()] == sap_info.n_linesearch_iterations[None]
        cost_tol = gs.qd_float(0.0)
        if is_first_pass:
            totals = func_chunks_total(
                i_b,
                0,
                sap_state.vertices_dofs_partials.shape[0],
                qd.static(VertexDofSum.SLOPE),
                0,
                sap_state,
                n_lanes=1,
                n_sums=3,
            )
            sap_state.envs.m[i_b] = totals[VertexDofSum.SLOPE]
            # The cost, whose round-off its changes are compared to
            energy = totals[VertexDofSum.ENERGY] + func_constraints_total(
                i_b, qd.static(ConstraintSum.ENERGY), sap_state
            )
            if qd.static(sap_config.has_fem):
                for i_k in range(sap_info.n_element_chunks[None]):
                    energy = energy + sap_state.elements_partials[i_k, i_b]
            sap_state.envs.prev_energy[i_b] = energy
            # The solve stops once a Newton iteration decreases its cost by no more than its round-off: a tolerance on
            # its gradient below what the precision of the simulation resolves would otherwise run it to the cap.
            energy_decrease = sap_state.envs.energy_newton_prev[i_b] - energy
            sap_state.envs.energy_newton_prev[i_b] = energy
            cost_tol = sap_info.linesearch_cost_atol[None] + sap_info.linesearch_cost_rtol[None] * energy
            if energy_decrease < cost_tol:
                sap_state.envs.is_active[i_b] = False
                sap_state.envs.is_linesearch_active[i_b] = False
                continue
            sap_state.envs.d2ellA_dalpha2[i_b] = totals[VertexDofSum.D2ELLA]
        dell_dalpha = func_chunks_total(
            i_b,
            0,
            sap_state.vertices_dofs_partials.shape[0],
            qd.static(VertexDofSum.DELL),
            0,
            sap_state,
            n_lanes=1,
            n_sums=1,
        )[0]
        sap_state.envs.dell_dalpha[i_b] = dell_dalpha + func_constraints_total(
            i_b, qd.static(ConstraintSum.DELL), sap_state
        )
        sap_state.envs.d2ell_dalpha2[i_b] = sap_state.envs.d2ellA_dalpha2[i_b] + func_constraints_total(
            i_b, qd.static(ConstraintSum.D2ELL), sap_state
        )
        if is_first_pass:
            sap_state.envs.is_linesearch_active[i_b] = sap_state.envs.dell_dalpha[i_b] > 0.0
            # When the slope is round-off, the full step is taken and the line search ends
            if sap_state.envs.is_linesearch_active[i_b] and -sap_state.envs.m[i_b] < cost_tol:
                sap_state.envs.is_linesearch_active[i_b] = False
                sap_state.envs.step_size[i_b] = 1.0
                if qd.static(sap_config.has_fem):
                    for i_v in range(sap_state.fem_vertices.v.shape[0]):
                        sap_state.fem_vertices.v[i_v, i_b] = (
                            sap_state.fem_vertices.x_prev[i_v, i_b] + sap_state.fem_vertices.x[i_v, i_b]
                        )
                if qd.static(sap_config.has_rigid):
                    for i_d in range(sap_info.n_dofs[None]):
                        sap_state.rigid_dofs.v[i_d, i_b] = (
                            sap_state.rigid_dofs.x_prev[i_d, i_b] + sap_state.rigid_dofs.x[i_d, i_b]
                        )
            if sap_state.envs.is_linesearch_active[i_b]:
                max_step_size = sap_info.linesearch_max_step_size[None]
                sap_state.envs.dell_scale[i_b] = -sap_state.envs.m[i_b]
                sap_state.envs.step_size[i_b] = qd.min(
                    -sap_state.envs.m[i_b] / sap_state.envs.d2ell_dalpha2[i_b], max_step_size
                )
                sap_state.envs.alpha_min[i_b] = 0.0
                sap_state.envs.alpha_max[i_b] = max_step_size
                sap_state.envs.f_lower[i_b] = -1.0
                sap_state.envs.f_upper[i_b] = sap_state.envs.dell_dalpha[i_b] / sap_state.envs.dell_scale[i_b]
                sap_state.envs.alpha_tol[i_b] = sap_info.linesearch_ftol[None] * sap_state.envs.step_size[i_b]
                sap_state.envs.minus_dalpha[i_b] = sap_state.envs.alpha_min[i_b] - sap_state.envs.alpha_max[i_b]
                sap_state.envs.minus_dalpha_prev[i_b] = sap_state.envs.minus_dalpha[i_b]
                if qd.abs(sap_state.envs.f_lower[i_b]) < sap_info.linesearch_ftol[None]:
                    sap_state.envs.is_linesearch_active[i_b] = False
                    sap_state.envs.step_size[i_b] = sap_state.envs.alpha_min[i_b]
                if qd.abs(sap_state.envs.f_upper[i_b]) < sap_info.linesearch_ftol[None]:
                    sap_state.envs.is_linesearch_active[i_b] = False
                    sap_state.envs.step_size[i_b] = sap_state.envs.alpha_max[i_b]
        else:
            sap_state.envs.f[i_b] = sap_state.envs.dell_dalpha[i_b] / sap_state.envs.dell_scale[i_b]
            sap_state.envs.df[i_b] = sap_state.envs.d2ell_dalpha2[i_b] / sap_state.envs.dell_scale[i_b]
            if qd.math.sign(sap_state.envs.f[i_b]) != qd.math.sign(sap_state.envs.f_upper[i_b]):
                sap_state.envs.alpha_min[i_b] = sap_state.envs.step_size[i_b]
                sap_state.envs.f_lower[i_b] = sap_state.envs.f[i_b]
            else:
                sap_state.envs.alpha_max[i_b] = sap_state.envs.step_size[i_b]
                sap_state.envs.f_upper[i_b] = sap_state.envs.f[i_b]
            if qd.abs(sap_state.envs.f[i_b]) < sap_info.linesearch_ftol[None]:
                sap_state.envs.is_linesearch_active[i_b] = False
                continue

            is_newton_slow = 2.0 * qd.abs(sap_state.envs.f[i_b]) > qd.abs(
                sap_state.envs.minus_dalpha_prev[i_b] * sap_state.envs.df[i_b]
            )
            sap_state.envs.minus_dalpha_prev[i_b] = sap_state.envs.minus_dalpha[i_b]
            is_bisected = is_newton_slow
            if not is_newton_slow:
                # Newton
                sap_state.envs.minus_dalpha[i_b] = sap_state.envs.f[i_b] / sap_state.envs.df[i_b]
                sap_state.envs.step_size[i_b] = sap_state.envs.step_size[i_b] - sap_state.envs.minus_dalpha[i_b]
                is_bisected = (
                    sap_state.envs.step_size[i_b] <= sap_state.envs.alpha_min[i_b]
                    or sap_state.envs.step_size[i_b] >= sap_state.envs.alpha_max[i_b]
                )
            if is_bisected:
                sap_state.envs.minus_dalpha[i_b] = 0.5 * (sap_state.envs.alpha_min[i_b] - sap_state.envs.alpha_max[i_b])
                sap_state.envs.step_size[i_b] = sap_state.envs.alpha_min[i_b] - sap_state.envs.minus_dalpha[i_b]
            if qd.abs(sap_state.envs.minus_dalpha[i_b]) < sap_state.envs.alpha_tol[i_b]:
                sap_state.envs.is_linesearch_active[i_b] = False
        if sap_state.envs.is_linesearch_active[i_b]:
            linesearch_counter[()] = sap_state.loops.n_linesearch_next[()]


# ------------------------------------------------------------------------------------
# ------------------------------------ Handlers --------------------------------------
# ------------------------------------------------------------------------------------
# Every constraint of the solve, a contact pair or an equality constraint, runs the operations of the solve through the
# 'compute_solve_op' of its handler (see func_compute_constraints), the FEM vertices and the rigid links it acts on
# gathering its impulse in their fixed order (see func_gather_fem_vertex and func_compute_link_wrench).


@qd.data_oriented
class RigidConstraintHandler(RBC):
    """
    The joint equality constraints of the rigid bodies in the SAP coupler.

    The joint equalities of every environment take the same slots, those of environment i_b starting at i_b times their
    number, so that their order is the same on every run.
    """

    # The FEM vertices and the rigid links each constraint acts on, through the corners of its tets and its sides (see
    # BaseContactHandler.fem_corner_vertex and rigid_side_link).
    n_fem_corners = 0
    n_rigid_sides = 0

    def __init__(self, simulator: "Simulator", stiffness: float = 1e8, beta: float = 0.1) -> None:
        rigid_solver = simulator.rigid_solver
        self.stiffness = qd.field(gs.qd_float, shape=())
        self.stiffness[None] = stiffness
        # Near-rigid threshold of SAP: the regularization keeps the period of a constraint above beta time steps
        self.beta = qd.field(gs.qd_float, shape=())
        self.beta[None] = beta
        self.constraint_info_type = qd.types.struct(
            k=gs.qd_float,  # constraint stiffness
            R=gs.qd_float,  # Regularization
            R_inv=gs.qd_float,  # Inverse of R
            v_hat=gs.qd_float,  # Stablization velocity
            gamma=gs.qd_float,  # contact impulse
            G=gs.qd_float,  # Hessian matrix
            GJp=gs.qd_float,  # product of the Hessian with the Jacobian of the search direction
            dvc=gs.qd_float,  # change in constraint velocity
        )
        # TODO: Maybe support different constraints for each batch in the future.
        # For now all batches have the same constraints.
        joint_equalities_idx = [
            i_e for i_e, equality in enumerate(rigid_solver.equalities) if equality.type == gs.EQUALITY_TYPE.JOINT
        ]
        n_env_constraints = len(joint_equalities_idx)
        self.max_constraints = n_env_constraints * simulator._B
        self.key_bits = max(1, (max(self.max_constraints, 1) - 1).bit_length())
        self.n_constraints = qd.field(gs.qd_int, shape=())
        self.n_constraints[None] = self.max_constraints
        self.joint_equalities_idx = qd.field(gs.qd_int, shape=(max(n_env_constraints, 1),))
        if n_env_constraints > 0:
            self.joint_equalities_idx.from_numpy(np.array(joint_equalities_idx, dtype=gs.np_int))
        self.constraint_type = qd.types.struct(
            batch_idx=gs.qd_int,  # batch index
            i_eq=gs.qd_int,  # index of the equality constraint
            i_dof1=gs.qd_int,  # index of the first DOF in the constraint
            i_dof2=gs.qd_int,  # index of the second DOF in the constraint
            i_qpos1=gs.qd_int,  # index of the first joint position
            i_qpos2=gs.qd_int,  # index of the second joint position
            jac_dof2=gs.qd_float,  # Jacobian entry of the second DOF, the first one being 1
            constraint_info=self.constraint_info_type,
        )
        self.constraints = self.constraint_type.field(shape=(max(self.max_constraints, 1),))
        # Vectors of size one, as the rigid Jacobian solve shared with the contact handlers indexes each component
        self.Jt = qd.field(qd.types.vector(1, gs.qd_float), shape=(max(self.max_constraints, 1), rigid_solver.n_dofs_))
        self.M_inv_Jt = qd.field(
            qd.types.vector(1, gs.qd_float), shape=(max(self.max_constraints, 1), rigid_solver.n_dofs_)
        )

    @qd.func
    def envs_n_constraints(self, i_b: int):
        """Return the number of constraints of an environment."""
        return self.joint_equalities_idx.shape[0]

    @qd.func
    def constraint_env(self, i_c: int):
        """Return the environment of a constraint."""
        return self.constraints[i_c].batch_idx

    @qd.func
    def constraint_key(self, i_c: int):
        """Return the key of a constraint, unique among those of its environment (see kernel_order_constraints)."""
        return qd.u64(i_c)

    @qd.func
    def compute_regularization(
        self, entities_info: EntitiesInfo, equalities_info: EqualitiesInfo, rigid_info: RigidInfo, sap_info: SAPInfo
    ):
        # The constraint q1 - q1_0 = poly(q2 - q2_0) of the polynomial coefficients 'eq_data', linearized at the current
        # joint positions as the rigid constraint solver does, so its Jacobian is [1, -poly'(q2 - q2_0)].
        dt_inv = 1.0 / sap_info.substep_dt[None]
        constraint_info = qd.static(self.constraints.constraint_info)
        for i_c in range(self.n_constraints[None]):
            i_b = self.constraints[i_c].batch_idx
            i_e = self.constraints[i_c].i_eq
            i_qpos1 = self.constraints[i_c].i_qpos1
            i_qpos2 = self.constraints[i_c].i_qpos2
            pos = rigid_info.qpos[i_qpos1, i_b] - rigid_info.qpos0[i_qpos1, i_b]
            diff = rigid_info.qpos[i_qpos2, i_b] - rigid_info.qpos0[i_qpos2, i_b]
            g0, deriv = gu.joint_equality_residual(pos, diff, equalities_info.eq_data[i_e, i_b])
            i_dof2 = self.constraints[i_c].i_dof2
            self.constraints[i_c].jac_dof2 = -deriv
            self.Jt[i_c, i_dof2] = qd.Vector([-deriv])
            constraint_info[i_c].v_hat = -g0 * dt_inv

        # The mass matrix solved against the transposed Jacobian of every constraint, through its factor L D L^T
        M_inv_Jt = qd.static(self.M_inv_Jt)
        for i_c, i_e in qd.ndrange(self.n_constraints[None], entities_info.dof_start.shape[0]):
            i_b = self.constraints[i_c].batch_idx
            entity_dof_end = entities_info.dof_end[i_e]
            for i_d_ in range(entities_info.n_dofs[i_e]):
                i_d = entity_dof_end - i_d_ - 1
                M_inv_Jt[i_c, i_d][0] = self.Jt[i_c, i_d][0]
                for j_d in range(i_d + 1, entity_dof_end):
                    M_inv_Jt[i_c, i_d][0] = (
                        M_inv_Jt[i_c, i_d][0] - rigid_info.mass_mat_L[j_d, i_d, i_b] * M_inv_Jt[i_c, j_d][0]
                    )
        for i_c, i_d in qd.ndrange(self.n_constraints[None], sap_info.n_dofs[None]):
            i_b = self.constraints[i_c].batch_idx
            M_inv_Jt[i_c, i_d][0] = M_inv_Jt[i_c, i_d][0] * rigid_info.mass_mat_D_inv[i_d, i_b]
        for i_c, i_e in qd.ndrange(self.n_constraints[None], entities_info.dof_start.shape[0]):
            i_b = self.constraints[i_c].batch_idx
            entity_dof_start = entities_info.dof_start[i_e]
            entity_dof_end = entities_info.dof_end[i_e]
            for i_d in range(entity_dof_start, entity_dof_end):
                for j_d in range(entity_dof_start, i_d):
                    M_inv_Jt[i_c, i_d][0] = (
                        M_inv_Jt[i_c, i_d][0] - rigid_info.mass_mat_L[i_d, j_d, i_b] * M_inv_Jt[i_c, j_d][0]
                    )

        beta_factor = self.beta[None] ** 2 / (4.0 * qd.math.pi**2)
        for i_c in range(self.n_constraints[None]):
            # Delassus operator J @ M^-1 @ J^T
            i_dof1 = self.constraints[i_c].i_dof1
            i_dof2 = self.constraints[i_c].i_dof2
            W = self.M_inv_Jt[i_c, i_dof1][0] + self.constraints[i_c].jac_dof2 * self.M_inv_Jt[i_c, i_dof2][0]
            R = qd.max(beta_factor * W, 1.0 / (sap_info.dt2[None] * constraint_info[i_c].k))
            constraint_info[i_c].R = R
            constraint_info[i_c].R_inv = 1.0 / R

    @qd.func
    def compute_Jx(self, i_c: int, x: qd.Tensor):
        i_b = self.constraints[i_c].batch_idx
        i_dof1 = self.constraints[i_c].i_dof1
        i_dof2 = self.constraints[i_c].i_dof2
        return x[i_dof1, i_b] + self.constraints[i_c].jac_dof2 * x[i_dof2, i_b]

    @qd.func
    def compute_solve_op(
        self,
        i_c: int,
        dofs_state: DofsState,
        links_state: LinksState,
        sap_state: SAPState,
        fem_info: FEMInfo,
        links_info: LinksInfo,
        sap_info: SAPInfo,
        sap_config: qd.template(),
        op: qd.template(),
    ):
        """Run an operation of the solve on a constraint, and return its share of the sums of the operation.

        The shares are the cost and its first and second derivatives along the Newton step (see ConstraintOp).
        """
        constraint_info = qd.static(self.constraints.constraint_info)
        dell = gs.qd_float(0.0)
        d2ell = gs.qd_float(0.0)
        energy = gs.qd_float(0.0)
        if qd.static(op == ConstraintOp.MATRIX_VECTOR_PRODUCT):
            constraint_info[i_c].GJp = constraint_info[i_c].G * self.compute_Jx(i_c, sap_state.rigid_dofs.p)
        elif qd.static(op == ConstraintOp.SEARCH_DIRECTION):
            constraint_info[i_c].dvc = self.compute_Jx(i_c, sap_state.rigid_dofs.x)
        else:
            y = (constraint_info[i_c].v_hat - self.compute_Jx(i_c, sap_state.rigid_dofs.v)) * constraint_info[i_c].R_inv
            if qd.static(op == ConstraintOp.ENERGY):
                energy = 0.5 * y**2 * constraint_info[i_c].R
            else:
                constraint_info[i_c].gamma = y
                constraint_info[i_c].G = constraint_info[i_c].R_inv
                if qd.static(op == ConstraintOp.LINE_DERIVATIVES):
                    dvc = constraint_info[i_c].dvc
                    dell = -dvc * y
                    d2ell = dvc**2 * constraint_info[i_c].R_inv
        return dell, d2ell, energy


class ContactMode(IntEnum):
    STICK = 0
    SLIDE = 1
    NO_CONTACT = 2


@qd.data_oriented
class BaseContactHandler(RBC):
    """
    Base class for contact handling in SAPCoupler.

    This class provides a framework for managing contact pairs, computing gradients,
    and handling contact-related computations.
    """

    # Whether the handler collects contact candidates before its contact pairs
    has_candidates = False
    # The FEM vertices and the rigid links each contact pair acts on, see RigidConstraintHandler, and the FEM entities
    # whose rigid motions it acts on (see coarse_hessian).
    n_fem_corners = 0
    n_rigid_sides = 0
    n_coarse_sides = 1

    def __init__(self, simulator: "Simulator") -> None:
        self.n_contact_pairs = qd.field(gs.qd_int, shape=())
        # The contact pairs of each environment, past its share of the capacity on overflow (see func_add_slot)
        self.envs_n_contact_pairs = qd.field(gs.qd_int, shape=(simulator._B,))
        self.contact_info_type = qd.types.struct(
            k=gs.qd_float,  # contact stiffness
            phi0=gs.qd_float,  # initial signed distance
            Rn=gs.qd_float,  # Regularization for normal
            Rt=gs.qd_float,  # Regularization for tangential
            Rn_inv=gs.qd_float,  # Inverse of Rn
            Rt_inv=gs.qd_float,  # Inverse of Rt
            vn_hat=gs.qd_float,  # Stablization for normal velocity
            mu=gs.qd_float,  # friction coefficient
            mu_hat=gs.qd_float,  # friction coefficient regularized
            mu_factor=gs.qd_float,  # friction coefficient factor, 1/(1+mu_tilde**2)
            gamma=gs.qd_vec3,  # contact impulse
            G=gs.qd_mat3,  # Hessian matrix
            GJp=gs.qd_vec3,  # product of the Hessian with the Jacobian of the search direction
            dvc=gs.qd_vec3,  # velocity change at contact point, for exact line search
        )

    @property
    def n_constraints(self):
        """The number of contact pairs, each one constraint of the solve."""
        return self.n_contact_pairs

    @property
    def max_constraints(self):
        """The capacity of the contact pairs over all environments."""
        return self.max_contact_pairs

    @qd.func
    def envs_n_constraints(self, i_b: int):
        """Return the number of contact pairs of an environment, within its share of the capacity."""
        return qd.min(self.envs_n_contact_pairs[i_b], self.contact_pairs.shape[0] // self.envs_n_contact_pairs.shape[0])

    @qd.func
    def constraint_env(self, i_c: int):
        """Return the environment of a contact pair."""
        return self.contact_pairs[i_c].batch_idx

    @qd.func
    def constraint_key(self, i_c: int):
        """Return the key of a contact pair, unique among those of its environment (see kernel_order_constraints)."""
        return self.contact_pairs[i_c].key

    @qd.func
    def contact_frame(self, i_c: int):
        """Return the axes of the frame of a contact pair in its columns, the tangents then the normal."""
        return qd.Matrix.cols(
            [self.contact_pairs[i_c].tangent0, self.contact_pairs[i_c].tangent1, self.contact_pairs[i_c].normal]
        )

    @qd.func
    def compute_solve_op(
        self,
        i_c: int,
        dofs_state: DofsState,
        links_state: LinksState,
        sap_state: SAPState,
        fem_info: FEMInfo,
        links_info: LinksInfo,
        sap_info: SAPInfo,
        sap_config: qd.template(),
        op: qd.template(),
    ):
        """Run an operation of the solve on a contact pair, and return its share of the sums of the operation.

        The shares are the cost and its first and second derivatives along the Newton step (see ConstraintOp).
        """
        contact_info = qd.static(self.contact_pairs.contact_info)
        dell = gs.qd_float(0.0)
        d2ell = gs.qd_float(0.0)
        energy = gs.qd_float(0.0)
        if qd.static(op == ConstraintOp.MATRIX_VECTOR_PRODUCT):
            Jp = self.compute_Jx(
                i_c,
                sap_state.fem_vertices.p,
                sap_state.rigid_dofs.p,
                dofs_state,
                links_state,
                fem_info,
                links_info,
                sap_config,
            )
            contact_info[i_c].GJp = contact_info[i_c].G @ Jp
        elif qd.static(op == ConstraintOp.SEARCH_DIRECTION):
            contact_info[i_c].dvc = self.compute_Jx(
                i_c,
                sap_state.fem_vertices.x,
                sap_state.rigid_dofs.x,
                dofs_state,
                links_state,
                fem_info,
                links_info,
                sap_config,
            )
        else:
            vc = self.compute_Jx(
                i_c,
                sap_state.fem_vertices.v,
                sap_state.rigid_dofs.v,
                dofs_state,
                links_state,
                fem_info,
                links_info,
                sap_config,
            )
            cost, gamma, G = self.compute_contact_cost_gamma_G(i_c, vc, sap_info, sap_config)
            if qd.static(op == ConstraintOp.ENERGY):
                energy = cost
            else:
                contact_info[i_c].gamma = gamma
                contact_info[i_c].G = G
                if qd.static(op == ConstraintOp.LINE_DERIVATIVES):
                    dvc = contact_info[i_c].dvc
                    dell = -dvc.dot(gamma)
                    d2ell = dvc.dot(G @ dvc)
        return dell, d2ell, energy

    @qd.func
    def compute_regularization(
        self,
        fem_state: FEMState,
        dofs_state: DofsState,
        links_state: LinksState,
        sap_state: SAPState,
        fem_info: FEMInfo,
        links_info: LinksInfo,
        rigid_info: RigidInfo,
        sap_info: SAPInfo,
        sap_config: qd.template(),
    ):
        """
        Compute the regularization of every contact pair from its Delassus operator.
        """
        contact_info = qd.static(self.contact_pairs.contact_info)
        dt = sap_info.substep_dt[None]
        taud = sap_info.taud[None]
        for i_p in range(self.n_contact_pairs[None]):
            W = self.compute_delassus(
                i_p, fem_state, dofs_state, links_state, fem_info, links_info, rigid_info, sap_config
            )
            w_rms = W.norm() / 3.0
            Rn = qd.max(sap_info.beta_factor[None] * w_rms, 1.0 / (dt * contact_info[i_p].k * (dt + taud)))
            Rt = sap_info.sigma[None] * w_rms
            contact_info[i_p].Rn = Rn
            contact_info[i_p].Rt = Rt
            contact_info[i_p].Rn_inv = 1.0 / Rn
            contact_info[i_p].Rt_inv = 1.0 / Rt
            contact_info[i_p].vn_hat = -contact_info[i_p].phi0 / (dt + taud)
            contact_info[i_p].mu_hat = contact_info[i_p].mu * Rt / Rn
            contact_info[i_p].mu_factor = 1.0 / (1.0 + contact_info[i_p].mu * contact_info[i_p].mu_hat)

    @qd.func
    def compute_contact_cost_gamma_G(
        self, i_p: int, vc: qd.types.vector(3), sap_info: SAPInfo, sap_config: qd.template()
    ):
        """
        Return the cost of a contact pair at a contact velocity, its impulse and its Hessian.

        The pair is i_p and the velocity `vc`, which holds the tangential velocities then the normal one, positive when
        the sides move apart. The normal impulse is the projection of the regularized normal velocity error on the
        positive half-line, and the impulse the projection of the regularized velocity error on the friction cone, as in
        SAP.
        """
        contact_info = qd.static(self.contact_pairs.contact_info)
        cost = gs.qd_float(0.0)
        gamma = qd.Vector.zero(gs.qd_float, 3)
        G = qd.Matrix.zero(gs.qd_float, 3, 3)
        normal = qd.Vector([0.0, 0.0, 1.0], dt=gs.qd_float)
        Pn = normal.outer_product(normal)
        R = qd.Vector([contact_info[i_p].Rt, contact_info[i_p].Rt, contact_info[i_p].Rn])
        R_inv = qd.Vector([contact_info[i_p].Rt_inv, contact_info[i_p].Rt_inv, contact_info[i_p].Rn_inv])
        y = (qd.Vector([0.0, 0.0, contact_info[i_p].vn_hat]) - vc) * R_inv
        yr = y[:2].norm(sap_info.precision[None])
        yn = y[2]
        t_hat = y[:2] / yr
        mu = contact_info[i_p].mu
        mu_hat = contact_info[i_p].mu_hat
        mu_factor = contact_info[i_p].mu_factor
        # Stiction, inside the friction cone
        if yr <= mu * yn:
            gamma = y
            G = R_inv[0] * qd.Matrix.identity(gs.qd_float, 3) + (R_inv[2] - R_inv[0]) * Pn
        # Sliding, on the boundary of the friction cone, the other cases being out of contact
        elif -mu_hat * yr < yn and yn < yr / mu:
            gn = (yn + mu_hat * yr) * mu_factor
            gt = mu * gn * t_hat
            gamma = qd.Vector([gt[0], gt[1], gn])
            P = t_hat.outer_product(t_hat)
            Pperp = qd.Matrix.identity(gs.qd_float, 2) - P
            G[:2, :2] = mu * (gn / yr * Pperp + mu_hat * mu_factor * P) * R_inv[0]
            G[:2, 2] = mu * mu_factor * t_hat * R_inv[2]
            G[2, :2] = mu_hat * mu_factor * t_hat * R_inv[0]
            G[2, 2] = mu_factor * R_inv[2]
        cost = 0.5 * gamma.dot(R * gamma)
        return cost, gamma, G


@qd.data_oriented
class RigidContactHandler(BaseContactHandler):
    n_rigid_sides = 1

    @qd.func
    def rigid_side_link(self, i_c: int, i_side: int):
        """Return the link at a side of a contact pair."""
        return self.contact_pairs[i_c].link_idx

    @qd.func
    def rigid_side_sign(self, i_side: int):
        """Return the sign of the velocity of the link at a side in the velocity of its contact pairs."""
        return 1.0

    @qd.func
    def contact_frame(self, i_c: int):
        """Return the identity, the velocity of a contact with the floor holding its world axes, the normal last."""
        return qd.Matrix.identity(gs.qd_float, 3)

    @qd.func
    def compute_delassus(
        self,
        i_p: int,
        fem_state: FEMState,
        dofs_state: DofsState,
        links_state: LinksState,
        fem_info: FEMInfo,
        links_info: LinksInfo,
        rigid_info: RigidInfo,
        sap_config: qd.template(),
    ):
        pairs = qd.static(self.contact_pairs)
        return func_link_point_delassus(
            pairs[i_p].link_idx,
            -1,
            pairs[i_p].batch_idx,
            pairs[i_p].contact_pos,
            dofs_state,
            links_state,
            links_info,
            rigid_info,
            sap_config,
        )

    @qd.func
    def compute_Jx(
        self,
        i_p: int,
        x_fem: qd.Tensor,
        x_rigid: qd.Tensor,
        dofs_state: DofsState,
        links_state: LinksState,
        fem_info: FEMInfo,
        links_info: LinksInfo,
        sap_config: qd.template(),
    ):
        pairs = qd.static(self.contact_pairs)
        return func_link_point_jacobian_product(
            pairs[i_p].link_idx,
            pairs[i_p].batch_idx,
            pairs[i_p].contact_pos,
            x_rigid,
            dofs_state,
            links_state,
            links_info,
            sap_config,
        )


@qd.func
def accumulate_area_centroid(
    polygon_vertices, i: int, total_area: qd.template(), total_area_weighted_centroid: qd.template()
):
    e1 = polygon_vertices[:, i - 1] - polygon_vertices[:, 0]
    e2 = polygon_vertices[:, i] - polygon_vertices[:, 0]
    area = 0.5 * e1.cross(e2).norm()
    total_area += area
    total_area_weighted_centroid += (
        area * (polygon_vertices[:, 0] + polygon_vertices[:, i - 1] + polygon_vertices[:, i]) / 3.0
    )


@qd.func
def func_clip_polygon_by_tet(
    polygon_vertices: qd.template(),
    polygon_n_vertices: qd.template(),
    tet_vertices: qd.types.matrix(3, 4),
):
    """
    Clip a convex polygon by the four halfspaces of the faces of a tet, in place, and return whether at least three
    vertices are left.

    The faces take the outward normals of a tet of positive orientation. The polygon holds at most as many vertices as
    the columns of `polygon_vertices`.
    """
    normal_signs = qd.Vector([1.0, -1.0, 1.0, -1.0], dt=gs.qd_float)
    clipped_vertices = polygon_vertices * 0.0
    clipped_n_vertices = gs.qd_int(0)
    distances = qd.Vector.zero(gs.qd_float, polygon_vertices.m)
    is_valid = True
    for face in range(4):
        if is_valid:
            clipped_n_vertices = 0
            x = tet_vertices[:, (face + 1) % 4]
            normal = (tet_vertices[:, (face + 2) % 4] - x).cross(tet_vertices[:, (face + 3) % 4] - x) * normal_signs[
                face
            ]
            normal = normal / normal.norm()
            for i in range(polygon_n_vertices):
                distances[i] = (polygon_vertices[:, i] - x).dot(normal)
            for i in range(polygon_n_vertices):
                j = (i + 1) % polygon_n_vertices
                if distances[i] <= 0.0:
                    clipped_vertices[:, clipped_n_vertices] = polygon_vertices[:, i]
                    clipped_n_vertices = clipped_n_vertices + 1
                if distances[i] * distances[j] < 0.0:
                    wa = distances[j] / (distances[j] - distances[i])
                    clipped_vertices[:, clipped_n_vertices] = (
                        wa * polygon_vertices[:, i] + (1.0 - wa) * polygon_vertices[:, j]
                    )
                    clipped_n_vertices = clipped_n_vertices + 1
            polygon_n_vertices = clipped_n_vertices
            polygon_vertices = clipped_vertices
            is_valid = polygon_n_vertices >= 3
    return is_valid


@qd.data_oriented
class RigidRigidContactHandler(RigidContactHandler):
    n_rigid_sides = 2

    @qd.func
    def rigid_side_link(self, i_c: int, i_side: int):
        i_l = self.contact_pairs[i_c].link_idx0
        if i_side == 1:
            i_l = self.contact_pairs[i_c].link_idx1
        return i_l

    @qd.func
    def rigid_side_sign(self, i_side: int):
        return 1.0 - 2.0 * i_side

    @qd.func
    def contact_frame(self, i_c: int):
        return qd.Matrix.cols(
            [self.contact_pairs[i_c].tangent0, self.contact_pairs[i_c].tangent1, self.contact_pairs[i_c].normal]
        )

    @qd.func
    def compute_delassus(
        self,
        i_p: int,
        fem_state: FEMState,
        dofs_state: DofsState,
        links_state: LinksState,
        fem_info: FEMInfo,
        links_info: LinksInfo,
        rigid_info: RigidInfo,
        sap_config: qd.template(),
    ):
        pairs = qd.static(self.contact_pairs)
        world = self.contact_frame(i_p)
        W = func_link_point_delassus(
            pairs[i_p].link_idx0,
            pairs[i_p].link_idx1,
            pairs[i_p].batch_idx,
            pairs[i_p].contact_pos,
            dofs_state,
            links_state,
            links_info,
            rigid_info,
            sap_config,
        )
        return world.transpose() @ W @ world

    @qd.func
    def compute_Jx(
        self,
        i_p: int,
        x_fem: qd.Tensor,
        x_rigid: qd.Tensor,
        dofs_state: DofsState,
        links_state: LinksState,
        fem_info: FEMInfo,
        links_info: LinksInfo,
        sap_config: qd.template(),
    ):
        pairs = qd.static(self.contact_pairs)
        i_b = pairs[i_p].batch_idx
        Jx = qd.Vector.zero(gs.qd_float, 3)
        for i_s in range(2):
            i_l = pairs[i_p].link_idx0
            sign = 1.0
            if i_s == 1:
                i_l = pairs[i_p].link_idx1
                sign = -1.0
            Jx = Jx + sign * func_link_point_jacobian_product(
                i_l, i_b, pairs[i_p].contact_pos, x_rigid, dofs_state, links_state, links_info, sap_config
            )
        return self.contact_frame(i_p).transpose() @ Jx


@qd.data_oriented
class FEMContactHandler(BaseContactHandler):
    @qd.func
    def fem_delassus_vertex(self, i_v: int, i_b: int, w: float, fem_state: FEMState, fem_info: FEMInfo):
        """Return the share of a weighted FEM vertex in the Delassus operator of a contact, J A^-1 J^T.

        The weight of the vertex is `w`.
        """
        return w**2 * fem_info.dt2_inv[None] * fem_state.vertices_solve.prec[i_v, i_b]


@qd.data_oriented
class FEMFloorTetContactHandler(FEMContactHandler):
    """
    Class for handling contact between a tetrahedral mesh and a floor in a simulation using hydroelastic model.

    This class extends the BaseContact class and provides methods for detecting contact
    between the tetrahedral elements and the floor, computing contact pairs, and managing
    contact-related computations.
    """

    has_candidates = True
    n_fem_corners = 4

    def __init__(self, simulator: "Simulator", options: SAPCouplerOptions) -> None:
        super().__init__(simulator)
        fem_solver = simulator.fem_solver
        self.name = "FEMFloorTetContactHandler"
        # The key of a pair is its tet
        self.key_bits = max(1, fem_solver.n_elements - 1).bit_length()
        self.contact_candidate_type = qd.types.struct(
            batch_idx=gs.qd_int,  # batch index
            geom_idx=gs.qd_int,  # index of the FEM element
            intersection_code=gs.qd_int,  # intersection code for the element
            distance=gs.qd_vec4,  # distance vector for the element
        )
        self.n_contact_candidates = qd.field(gs.qd_int, shape=())
        self.envs_n_contact_candidates = qd.field(gs.qd_int, shape=(simulator._B,))
        # Each surface tet crosses the floor once at most, each crossing giving one contact at most
        n_contacts, n_candidates = get_contact_capacities(
            options, fem_solver.n_surface_elements, fem_solver.n_surface_elements
        )
        self.max_contact_candidates = n_candidates * simulator._B
        self.contact_candidates = self.contact_candidate_type.field(shape=(max(self.max_contact_candidates, 1),))

        self.contact_pair_type = qd.types.struct(
            batch_idx=gs.qd_int,  # batch index
            key=qd.u64,  # key unique among the pairs of the environment (see kernel_order_constraints)
            geom_idx=gs.qd_int,  # index of the FEM element
            barycentric=gs.qd_vec4,  # barycentric coordinates of the contact point
            contact_pos=gs.qd_vec3,  # contact position
            contact_info=self.contact_info_type,
        )
        self.max_contact_pairs = n_contacts * simulator._B
        self.contact_pairs = self.contact_pair_type.field(shape=(max(self.max_contact_pairs, 1),))

    @qd.func
    def detection(
        self,
        fem_state: FEMState,
        free_verts_state: VertsState,
        fixed_verts_state: VertsState,
        geoms_state: GeomsState,
        sap_state: SAPState,
        contact_queries_state: SAPContactQueriesState,
        fem_info: FEMInfo,
        links_info: LinksInfo,
        verts_info: VertsInfo,
        faces_info: FacesInfo,
        geoms_info: GeomsInfo,
        sap_info: SAPInfo,
        rigid_config: qd.template(),
        errno: qd.Tensor,
    ):
        n_envs = self.envs_n_contact_candidates.shape[0]
        self.n_contact_candidates[None] = 0
        # TODO Check surface element only instead of all elements
        for i_e, i_b in qd.ndrange(fem_info.elements.V.shape[0], n_envs, axes=qd.static(eoa(geoms_state.pos))):
            i_g = sap_info.ground_geom_idx[None]
            ground_pos = geoms_state.pos[i_g, i_b]
            intersection_code = qd.int32(0)
            distance = qd.Vector.zero(gs.qd_float, 4)
            for i in qd.static(range(4)):
                i_v = fem_info.elements.el2v[i_e][i]
                pos_v = fem_state.vertices.pos[i_v, i_b]
                distance[i] = pos_v.z - ground_pos.z
                if distance[i] > 0.0:
                    intersection_code |= 1 << i

            # check if the element intersect with the floor
            if intersection_code != 0 and intersection_code != 15:
                i_c = func_add_slot(
                    i_b,
                    self.contact_candidates.shape[0] // n_envs,
                    self.envs_n_contact_candidates,
                    self.n_contact_candidates,
                    code=SAPErrorCode.OVERFLOW_CANDIDATE_CONTACTS,
                    errno=errno,
                )
                if i_c >= 0:
                    self.contact_candidates[i_c].batch_idx = i_b
                    self.contact_candidates[i_c].geom_idx = i_e
                    self.contact_candidates[i_c].intersection_code = intersection_code
                    self.contact_candidates[i_c].distance = distance

        contact_info = qd.static(self.contact_pairs.contact_info)
        self.n_contact_pairs[None] = 0
        for i_c in range(self.n_contact_candidates[None]):
            candidate = self.contact_candidates[i_c]
            i_b = candidate.batch_idx
            i_e = candidate.geom_idx
            intersection_code = candidate.intersection_code
            intersected_edges = sap_info.marching_tets_edge_table[intersection_code]

            tet_vertices = qd.Matrix.zero(gs.qd_float, 3, 4)  # 4 vertices
            tet_pressures = qd.Vector.zero(gs.qd_float, 4)  # pressures at the vertices
            for i in qd.static(range(4)):
                i_v = fem_info.elements.el2v[i_e][i]
                tet_vertices[:, i] = fem_state.vertices.pos[i_v, i_b]
                tet_pressures[i] = sap_info.fem_pressure[i_v]

            polygon_vertices = qd.Matrix.zero(gs.qd_float, 3, 4)  # 3 or 4 vertices
            total_area = gs.qd_float(0.0)
            total_area_weighted_centroid = qd.Vector.zero(gs.qd_float, 3)
            for i in qd.static(range(4)):
                if intersected_edges[i] >= 0:
                    edge = sap_info.tet_edges[intersected_edges[i]]
                    pos_v0 = tet_vertices[:, edge[0]]
                    pos_v1 = tet_vertices[:, edge[1]]
                    d_v0 = candidate.distance[edge[0]]
                    d_v1 = candidate.distance[edge[1]]
                    t = d_v0 / (d_v0 - d_v1)
                    polygon_vertices[:, i] = pos_v0 + t * (pos_v1 - pos_v0)

                    # Compute triangle area and centroid
                    if qd.static(i >= 2):
                        accumulate_area_centroid(polygon_vertices, i, total_area, total_area_weighted_centroid)
            if total_area < sap_info.contact_area_min[None]:
                continue
            centroid = total_area_weighted_centroid / total_area

            # Compute barycentric coordinates
            barycentric = tet_barycentric(centroid, tet_vertices)
            pressure = barycentric.dot(tet_pressures)

            deformable_g = sap_info.hydroelastic_stiffness[None]
            grad = sap_state.fem_pressure_gradient[i_e, i_b]
            rigid_g = grad.z
            if rigid_g <= sap_info.contact_gradient_rtol[None] * grad.norm():
                continue
            # TODO: The published hydroelastic model treats the rigid side as infinitely stiff (g = FEM gradient alone).
            # Also, 'hydroelastic_stiffness' is a pressure gradient here but a modulus in the rigid pressure fields.
            g = 1.0 / (1.0 / deformable_g + 1.0 / rigid_g)  # harmonic average
            rigid_k = total_area * g
            rigid_phi0 = -pressure / g
            if rigid_k < sap_info.contact_stiffness_min[None] or rigid_phi0 > sap_info.contact_separation_max[None]:
                continue
            i_p = func_add_slot(
                i_b,
                self.contact_pairs.shape[0] // n_envs,
                self.envs_n_contact_pairs,
                self.n_contact_pairs,
                code=SAPErrorCode.OVERFLOW_CONTACTS,
                errno=errno,
            )
            if i_p >= 0:
                self.contact_pairs[i_p].batch_idx = i_b
                self.contact_pairs[i_p].key = qd.u64(i_e)
                self.contact_pairs[i_p].geom_idx = i_e
                self.contact_pairs[i_p].barycentric = barycentric
                self.contact_pairs[i_p].contact_pos = centroid
                contact_info[i_p].k = rigid_k
                contact_info[i_p].phi0 = rigid_phi0
                contact_info[i_p].mu = fem_info.elements.friction_mu[i_e]

    @qd.func
    def contact_frame(self, i_c: int):
        """Return the identity, the velocity of a contact with the floor holding its world axes, the normal last."""
        return qd.Matrix.identity(gs.qd_float, 3)

    @qd.func
    def fem_corner_vertex(self, i_c: int, i_corner: int, fem_info: FEMInfo):
        """Return the FEM vertex at a corner of a contact pair."""
        i_e = self.contact_pairs[i_c].geom_idx
        return fem_info.elements.el2v[i_e][i_corner]

    @qd.func
    def fem_corner_weight(self, i_c: int, i_corner: int):
        """Return the weight of the velocity of a corner in the velocity of its contact pair."""
        return self.contact_pairs[i_c].barycentric[i_corner]

    @qd.func
    def compute_Jx(
        self,
        i_p: int,
        x_fem: qd.Tensor,
        x_rigid: qd.Tensor,
        dofs_state: DofsState,
        links_state: LinksState,
        fem_info: FEMInfo,
        links_info: LinksInfo,
        sap_config: qd.template(),
    ):
        i_b = self.contact_pairs[i_p].batch_idx
        i_e = self.contact_pairs[i_p].geom_idx
        Jx = qd.Vector.zero(gs.qd_float, 3)
        for i in qd.static(range(4)):
            i_v = fem_info.elements.el2v[i_e][i]
            Jx = Jx + self.contact_pairs[i_p].barycentric[i] * x_fem[i_v, i_b]
        return Jx

    @qd.func
    def coarse_hessian(self, i_p: int, i_side: int, sap_state: SAPState, fem_info: FEMInfo, sap_info: SAPInfo):
        """Return the FEM entity and the Hessian restricted to its rigid motions of a contact pair."""
        i_e = self.contact_pairs[i_p].geom_idx
        i_v = fem_info.elements.el2v[i_e][0]
        return func_fem_coarse_contact_hessian(
            i_v,
            self.contact_pairs[i_p].batch_idx,
            self.contact_pairs[i_p].contact_pos,
            qd.Matrix.identity(gs.qd_float, 3),
            self.contact_pairs.contact_info.G[i_p],
            sap_state,
            sap_info,
        )

    @qd.func
    def compute_delassus(
        self,
        i_p: int,
        fem_state: FEMState,
        dofs_state: DofsState,
        links_state: LinksState,
        fem_info: FEMInfo,
        links_info: LinksInfo,
        rigid_info: RigidInfo,
        sap_config: qd.template(),
    ):
        i_b = self.contact_pairs[i_p].batch_idx
        i_e = self.contact_pairs[i_p].geom_idx
        # With floor, J is Identity times the barycentric coordinates
        W = qd.Matrix.zero(gs.qd_float, 3, 3)
        for i in qd.static(range(4)):
            i_v = fem_info.elements.el2v[i_e][i]
            W = W + self.fem_delassus_vertex(i_v, i_b, self.contact_pairs[i_p].barycentric[i], fem_state, fem_info)
        return W


@qd.data_oriented
class FEMSelfTetContactHandler(FEMContactHandler):
    """
    Class for handling self-contact between tetrahedral elements in a simulation using hydroelastic model.

    This class extends the FEMContact class and provides methods for detecting self-contact
    between tetrahedral elements, computing contact pairs, and managing contact-related computations.
    """

    has_candidates = True
    n_fem_corners = 8
    n_coarse_sides = 2

    def __init__(self, simulator: "Simulator", options: SAPCouplerOptions) -> None:
        super().__init__(simulator)
        fem_solver = simulator.fem_solver
        self.name = "FEMSelfTetContactHandler"
        # The key of a pair is its two tets
        self.element_bits = max(1, fem_solver.n_elements - 1).bit_length()
        self.key_bits = 2 * self.element_bits
        self.contact_candidate_type = qd.types.struct(
            batch_idx=gs.qd_int,  # batch index
            geom_idx0=gs.qd_int,  # index of the FEM element0
            intersection_code0=gs.qd_int,  # intersection code for element0
            geom_idx1=gs.qd_int,  # index of the FEM element1
            normal=gs.qd_vec3,  # contact plane normal
            x=gs.qd_vec3,  # a point on the contact plane
            distance0=gs.qd_vec4,  # distance vector for element0
        )
        self.n_contact_candidates = qd.field(gs.qd_int, shape=())
        self.envs_n_contact_candidates = qd.field(gs.qd_int, shape=(simulator._B,))
        # Each pair of surface tets gives one candidate and one contact at most
        n_pairs_max = fem_solver.n_surface_elements * (fem_solver.n_surface_elements - 1) // 2
        n_contacts, n_candidates = get_contact_capacities(options, n_pairs_max, n_pairs_max)
        self.max_contact_candidates = n_candidates * simulator._B
        self.contact_candidates = self.contact_candidate_type.field(shape=(max(self.max_contact_candidates, 1),))

        self.contact_pair_type = qd.types.struct(
            batch_idx=gs.qd_int,  # batch index
            key=qd.u64,  # key unique among the pairs of the environment (see kernel_order_constraints)
            normal=gs.qd_vec3,  # contact plane normal
            tangent0=gs.qd_vec3,  # contact plane tangent0
            tangent1=gs.qd_vec3,  # contact plane tangent1
            geom_idx0=gs.qd_int,  # index of the FEM element0
            geom_idx1=gs.qd_int,  # index of the FEM element1
            barycentric0=gs.qd_vec4,  # barycentric coordinates of the contact point in tet 0
            barycentric1=gs.qd_vec4,  # barycentric coordinates of the contact point in tet 1
            contact_pos=gs.qd_vec3,  # contact position
            contact_info=self.contact_info_type,
        )
        self.max_contact_pairs = n_contacts * simulator._B
        self.contact_pairs = self.contact_pair_type.field(shape=(max(self.max_contact_pairs, 1),))

    @qd.func
    def detection(
        self,
        fem_state: FEMState,
        free_verts_state: VertsState,
        fixed_verts_state: VertsState,
        geoms_state: GeomsState,
        sap_state: SAPState,
        contact_queries_state: SAPContactQueriesState,
        fem_info: FEMInfo,
        links_info: LinksInfo,
        verts_info: VertsInfo,
        faces_info: FacesInfo,
        geoms_info: GeomsInfo,
        sap_info: SAPInfo,
        rigid_config: qd.template(),
        errno: qd.Tensor,
    ):
        n_envs = self.envs_n_contact_candidates.shape[0]
        contact_queries_state.fem_self.results.count[0] = 0
        func_bvh_query_leaves(
            contact_queries_state.fem_self,
            func_filter_fem_surface_tets,
            qd.static((fem_info.surface_elements, fem_info.elements.el2v)),
        )
        func_flag_query_overflow(contact_queries_state.fem_self.results, errno)

        # The candidates, the pairs of tets crossing their equal-pressure plane
        self.n_contact_candidates[None] = 0
        for i_r in range(contact_queries_state.fem_self.results.count[0]):
            i_b, i_sa, i_sq = contact_queries_state.fem_self.results.triplets[i_r]
            i_a = fem_info.surface_elements[i_sa]
            i_q = fem_info.surface_elements[i_sq]
            i_v0 = fem_info.elements.el2v[i_a][0]
            i_v1 = fem_info.elements.el2v[i_q][0]
            x0 = fem_state.vertices.pos[i_v0, i_b]
            x1 = fem_state.vertices.pos[i_v1, i_b]
            p0 = sap_info.fem_pressure[i_v0]
            p1 = sap_info.fem_pressure[i_v1]
            g0 = sap_state.fem_pressure_gradient[i_a, i_b]
            g1 = sap_state.fem_pressure_gradient[i_q, i_b]
            g0_norm = g0.norm()
            g1_norm = g1.norm()
            if g0_norm == 0.0 or g1_norm == 0.0:
                continue
            # Calculate the isosurface, i.e. equal pressure plane defined by x and normal
            # Solve for p0 + g0.dot(x - x0) = p1 + g1.dot(x - x1)
            normal = g0 - g1
            magnitude = normal.norm()
            if magnitude <= sap_info.contact_gradient_rtol[None] * (g0_norm + g1_norm):
                continue
            normal = normal / magnitude
            b = p1 - p0 - g1.dot(x1) + g0.dot(x0)
            x = b / magnitude * normal
            # Check that the normal is pointing along g0 and against g1 (see COS_ANGLE_THRESHOLD)
            threshold = sap_info.cos_angle_threshold[None]
            if normal.dot(g0) < threshold * g0_norm or normal.dot(g1) > -threshold * g1_norm:
                continue
            intersection_code0 = qd.int32(0)
            distance0 = qd.Vector.zero(gs.qd_float, 4)
            intersection_code1 = qd.int32(0)
            for i in qd.static(range(4)):
                i_v = fem_info.elements.el2v[i_a][i]
                distance0[i] = (fem_state.vertices.pos[i_v, i_b] - x).dot(normal)  # signed distance
                if distance0[i] > 0.0:
                    intersection_code0 |= 1 << i
            for i in qd.static(range(4)):
                i_v = fem_info.elements.el2v[i_q][i]
                if (fem_state.vertices.pos[i_v, i_b] - x).dot(normal) > 0.0:
                    intersection_code1 |= 1 << i
            # Fast check for whether both tets intersect with the plane
            if (
                intersection_code0 == 0
                or intersection_code1 == 0
                or intersection_code0 == 15
                or intersection_code1 == 15
            ):
                continue
            i_c = func_add_slot(
                i_b,
                self.contact_candidates.shape[0] // n_envs,
                self.envs_n_contact_candidates,
                self.n_contact_candidates,
                code=SAPErrorCode.OVERFLOW_CANDIDATE_CONTACTS,
                errno=errno,
            )
            if i_c >= 0:
                self.contact_candidates[i_c].batch_idx = i_b
                self.contact_candidates[i_c].normal = normal
                self.contact_candidates[i_c].x = x
                self.contact_candidates[i_c].geom_idx0 = i_a
                self.contact_candidates[i_c].intersection_code0 = intersection_code0
                self.contact_candidates[i_c].distance0 = distance0
                self.contact_candidates[i_c].geom_idx1 = i_q

        # The pairs, the polygon of the plane in tet 0 clipped by tet 1. Intersection code reference:
        # https://github.com/RobotLocomotion/drake/blob/8c3a249184ed09f0faab3c678536d66d732809ce/geometry/proximity/field_intersection.cc#L87
        contact_info = qd.static(self.contact_pairs.contact_info)
        self.n_contact_pairs[None] = 0
        for i_c in range(self.n_contact_candidates[None]):
            i_b = self.contact_candidates[i_c].batch_idx
            i_e0 = self.contact_candidates[i_c].geom_idx0
            i_e1 = self.contact_candidates[i_c].geom_idx1
            distance0 = self.contact_candidates[i_c].distance0
            intersected_edges0 = sap_info.marching_tets_edge_table[self.contact_candidates[i_c].intersection_code0]

            tet_vertices0 = qd.Matrix.zero(gs.qd_float, 3, 4)  # 4 vertices of tet 0
            tet_pressures0 = qd.Vector.zero(gs.qd_float, 4)  # pressures at the vertices of tet 0
            tet_vertices1 = qd.Matrix.zero(gs.qd_float, 3, 4)  # 4 vertices of tet 1
            for i in qd.static(range(4)):
                i_v = fem_info.elements.el2v[i_e0][i]
                tet_vertices0[:, i] = fem_state.vertices.pos[i_v, i_b]
                tet_pressures0[i] = sap_info.fem_pressure[i_v]
            for i in qd.static(range(4)):
                i_v = fem_info.elements.el2v[i_e1][i]
                tet_vertices1[:, i] = fem_state.vertices.pos[i_v, i_b]

            polygon_vertices = qd.Matrix.zero(gs.qd_float, 3, 8)  # maximum 8 vertices
            polygon_n_vertices = gs.qd_int(0)
            for i in range(4):
                if intersected_edges0[i] >= 0:
                    edge = sap_info.tet_edges[intersected_edges0[i]]
                    pos_v0 = tet_vertices0[:, edge[0]]
                    pos_v1 = tet_vertices0[:, edge[1]]
                    t = distance0[edge[0]] / (distance0[edge[0]] - distance0[edge[1]])
                    polygon_vertices[:, polygon_n_vertices] = pos_v0 + t * (pos_v1 - pos_v0)
                    polygon_n_vertices = polygon_n_vertices + 1
            if not func_clip_polygon_by_tet(polygon_vertices, polygon_n_vertices, tet_vertices1):
                continue

            # compute centroid and area of the polygon
            total_area = gs.qd_float(0.0)
            total_area_weighted_centroid = qd.Vector.zero(gs.qd_float, 3)
            for i in range(2, polygon_n_vertices):
                accumulate_area_centroid(polygon_vertices, i, total_area, total_area_weighted_centroid)
            if total_area < sap_info.contact_area_min[None]:
                continue
            centroid = total_area_weighted_centroid / total_area
            barycentric0 = tet_barycentric(centroid, tet_vertices0)
            barycentric1 = tet_barycentric(centroid, tet_vertices1)
            normal = self.contact_candidates[i_c].normal
            tangent0 = polygon_vertices[:, 0] - centroid
            tangent0 = tangent0 / tangent0.norm()
            tangent1 = normal.cross(tangent0)

            pressure = barycentric0.dot(tet_pressures0)
            grad0 = sap_state.fem_pressure_gradient[i_e0, i_b]
            grad1 = sap_state.fem_pressure_gradient[i_e1, i_b]
            g0 = grad0.dot(normal)
            g1 = -grad1.dot(normal)
            g_rtol = sap_info.contact_gradient_rtol[None]
            # The stiffness of a contact is its area times the pressure gradients of both sides along its normal,
            # harmonically averaged, as for the other hydroelastic contacts. A side whose pressure does not increase
            # into its body along the normal has no stiffness to give.
            if g0 <= g_rtol * grad0.norm() or g1 <= g_rtol * grad1.norm():
                continue
            g = 1.0 / (1.0 / g0 + 1.0 / g1)  # harmonic average
            # FIXME This is an approximated value, different from Drake, which actually calculates the distance
            deformable_phi0 = -pressure / g
            if deformable_phi0 > sap_info.contact_separation_max[None]:
                continue

            i_p = func_add_slot(
                i_b,
                self.contact_pairs.shape[0] // n_envs,
                self.envs_n_contact_pairs,
                self.n_contact_pairs,
                code=SAPErrorCode.OVERFLOW_CONTACTS,
                errno=errno,
            )
            if i_p >= 0:
                self.contact_pairs[i_p].batch_idx = i_b
                self.contact_pairs[i_p].normal = normal
                self.contact_pairs[i_p].tangent0 = tangent0
                self.contact_pairs[i_p].tangent1 = tangent1
                self.contact_pairs[i_p].key = (qd.u64(i_e0) << qd.u64(self.element_bits)) | qd.u64(i_e1)
                self.contact_pairs[i_p].geom_idx0 = i_e0
                self.contact_pairs[i_p].geom_idx1 = i_e1
                self.contact_pairs[i_p].barycentric0 = barycentric0
                self.contact_pairs[i_p].barycentric1 = barycentric1
                self.contact_pairs[i_p].contact_pos = centroid
                contact_info[i_p].k = total_area * g
                contact_info[i_p].phi0 = deformable_phi0
                contact_info[i_p].mu = qd.sqrt(
                    fem_info.elements.friction_mu[i_e0] * fem_info.elements.friction_mu[i_e1]
                )

    @qd.func
    def fem_corner_vertex(self, i_c: int, i_corner: int, fem_info: FEMInfo):
        """Return the FEM vertex at a corner of a contact pair, the corners of its second tet last."""
        i_e = self.contact_pairs[i_c].geom_idx0
        if i_corner >= 4:
            i_e = self.contact_pairs[i_c].geom_idx1
        return fem_info.elements.el2v[i_e][i_corner % 4]

    @qd.func
    def fem_corner_weight(self, i_c: int, i_corner: int):
        """Return the weight of the velocity of a corner in the velocity of its contact pair."""
        w = gs.qd_float(0.0)
        if i_corner < 4:
            w = self.contact_pairs[i_c].barycentric0[i_corner]
        else:
            w = -self.contact_pairs[i_c].barycentric1[i_corner - 4]
        return w

    @qd.func
    def compute_Jx(
        self,
        i_p: int,
        x_fem: qd.Tensor,
        x_rigid: qd.Tensor,
        dofs_state: DofsState,
        links_state: LinksState,
        fem_info: FEMInfo,
        links_info: LinksInfo,
        sap_config: qd.template(),
    ):
        i_b = self.contact_pairs[i_p].batch_idx
        i_e0 = self.contact_pairs[i_p].geom_idx0
        i_e1 = self.contact_pairs[i_p].geom_idx1
        Jx = qd.Vector.zero(gs.qd_float, 3)
        for i in qd.static(range(4)):
            i_v = fem_info.elements.el2v[i_e0][i]
            Jx = Jx + self.contact_pairs[i_p].barycentric0[i] * x_fem[i_v, i_b]
        for i in qd.static(range(4)):
            i_v = fem_info.elements.el2v[i_e1][i]
            Jx = Jx - self.contact_pairs[i_p].barycentric1[i] * x_fem[i_v, i_b]
        return self.contact_frame(i_p).transpose() @ Jx

    @qd.func
    def coarse_hessian(self, i_p: int, i_side: int, sap_state: SAPState, fem_info: FEMInfo, sap_info: SAPInfo):
        """
        Return the FEM entity of a side of a contact pair and the Hessian of the pair restricted to its rigid motions.

        A contact of an entity with itself leaves its rigid motions free, both sides moving with them, which the entity
        -1 stands for.
        """
        i_e0 = self.contact_pairs[i_p].geom_idx0
        i_e1 = self.contact_pairs[i_p].geom_idx1
        i_v0 = fem_info.elements.el2v[i_e0][0]
        i_v1 = fem_info.elements.el2v[i_e1][0]
        i_v = qd.select(i_side == 0, i_v0, i_v1)
        i_e, H = func_fem_coarse_contact_hessian(
            i_v,
            self.contact_pairs[i_p].batch_idx,
            self.contact_pairs[i_p].contact_pos,
            self.contact_frame(i_p),
            self.contact_pairs.contact_info.G[i_p],
            sap_state,
            sap_info,
        )
        i_entity0 = sap_info.fem_vertices_entity_idx[i_v0]
        i_entity1 = sap_info.fem_vertices_entity_idx[i_v1]
        if i_entity0 == i_entity1:
            i_e = -1
        return i_e, H

    @qd.func
    def compute_delassus(
        self,
        i_p: int,
        fem_state: FEMState,
        dofs_state: DofsState,
        links_state: LinksState,
        fem_info: FEMInfo,
        links_info: LinksInfo,
        rigid_info: RigidInfo,
        sap_config: qd.template(),
    ):
        i_b = self.contact_pairs[i_p].batch_idx
        i_e0 = self.contact_pairs[i_p].geom_idx0
        i_e1 = self.contact_pairs[i_p].geom_idx1
        W = qd.Matrix.zero(gs.qd_float, 3, 3)
        for i in qd.static(range(4)):
            i_v = fem_info.elements.el2v[i_e0][i]
            W = W + self.fem_delassus_vertex(i_v, i_b, self.contact_pairs[i_p].barycentric0[i], fem_state, fem_info)
        for i in qd.static(range(4)):
            i_v = fem_info.elements.el2v[i_e1][i]
            W = W + self.fem_delassus_vertex(i_v, i_b, self.contact_pairs[i_p].barycentric1[i], fem_state, fem_info)
        world = self.contact_frame(i_p)
        return world.transpose() @ W @ world


@qd.data_oriented
class FEMFloorVertContactHandler(FEMContactHandler):
    """
    Class for handling contact between tetrahedral elements and a floor in a simulation using point contact model.

    This class extends the FEMContact class and provides methods for detecting contact
    between the tetrahedral elements and the floor, computing contact pairs, and managing
    contact-related computations.
    """

    n_fem_corners = 1

    def __init__(self, simulator: "Simulator", options: SAPCouplerOptions) -> None:
        super().__init__(simulator)
        fem_solver = simulator.fem_solver
        self.name = "FEMFloorVertContactHandler"
        # The key of a pair is its vertex
        self.key_bits = max(1, fem_solver.n_vertices - 1).bit_length()
        self.contact_pair_type = qd.types.struct(
            batch_idx=gs.qd_int,  # batch index
            key=qd.u64,  # key unique among the pairs of the environment (see kernel_order_constraints)
            geom_idx=gs.qd_int,  # index of the vertex
            contact_pos=gs.qd_vec3,  # contact position
            contact_info=self.contact_info_type,
        )
        # Every surface vertex of an entity lying flat may touch the floor
        n_contacts, _ = get_contact_capacities(options, fem_solver.n_surface_elements, 0)
        self.max_contact_pairs = n_contacts * simulator._B
        self.contact_pairs = self.contact_pair_type.field(shape=(max(self.max_contact_pairs, 1),))

    @qd.func
    def detection(
        self,
        fem_state: FEMState,
        free_verts_state: VertsState,
        fixed_verts_state: VertsState,
        geoms_state: GeomsState,
        sap_state: SAPState,
        contact_queries_state: SAPContactQueriesState,
        fem_info: FEMInfo,
        links_info: LinksInfo,
        verts_info: VertsInfo,
        faces_info: FacesInfo,
        geoms_info: GeomsInfo,
        sap_info: SAPInfo,
        rigid_config: qd.template(),
        errno: qd.Tensor,
    ):
        n_envs = self.envs_n_contact_pairs.shape[0]
        contact_info = qd.static(self.contact_pairs.contact_info)
        self.n_contact_pairs[None] = 0
        for i_sv, i_b in qd.ndrange(fem_info.surface_vertices.shape[0], n_envs, axes=qd.static(eoa(geoms_state.pos))):
            i_v = fem_info.surface_vertices[i_sv]
            pos_v = fem_state.vertices.pos[i_v, i_b]
            i_g = sap_info.ground_geom_idx[None]
            distance = pos_v.z - geoms_state.pos[i_g, i_b].z
            if distance > 0.0:
                continue
            i_p = func_add_slot(
                i_b,
                self.contact_pairs.shape[0] // n_envs,
                self.envs_n_contact_pairs,
                self.n_contact_pairs,
                code=SAPErrorCode.OVERFLOW_CONTACTS,
                errno=errno,
            )
            if i_p >= 0:
                self.contact_pairs[i_p].batch_idx = i_b
                self.contact_pairs[i_p].key = qd.u64(i_v)
                self.contact_pairs[i_p].geom_idx = i_v
                self.contact_pairs[i_p].contact_pos = pos_v
                contact_info[i_p].k = sap_info.point_contact_stiffness[None] * fem_info.surface_vert_mass[i_sv]
                contact_info[i_p].phi0 = distance
                contact_info[i_p].mu = fem_info.vertices.friction_mu[i_v]

    @qd.func
    def contact_frame(self, i_c: int):
        """Return the identity, the velocity of a contact with the floor holding its world axes, the normal last."""
        return qd.Matrix.identity(gs.qd_float, 3)

    @qd.func
    def fem_corner_vertex(self, i_c: int, i_corner: int, fem_info: FEMInfo):
        """Return the FEM vertex of a contact pair."""
        return self.contact_pairs[i_c].geom_idx

    @qd.func
    def fem_corner_weight(self, i_c: int, i_corner: int):
        """Return the weight of the velocity of the vertex in the velocity of its contact pair."""
        return gs.qd_float(1.0)

    @qd.func
    def compute_Jx(
        self,
        i_p: int,
        x_fem: qd.Tensor,
        x_rigid: qd.Tensor,
        dofs_state: DofsState,
        links_state: LinksState,
        fem_info: FEMInfo,
        links_info: LinksInfo,
        sap_config: qd.template(),
    ):
        return x_fem[self.contact_pairs[i_p].geom_idx, self.contact_pairs[i_p].batch_idx]

    @qd.func
    def coarse_hessian(self, i_p: int, i_side: int, sap_state: SAPState, fem_info: FEMInfo, sap_info: SAPInfo):
        """Return the FEM entity and the Hessian restricted to its rigid motions of a contact pair."""
        return func_fem_coarse_contact_hessian(
            self.contact_pairs[i_p].geom_idx,
            self.contact_pairs[i_p].batch_idx,
            self.contact_pairs[i_p].contact_pos,
            qd.Matrix.identity(gs.qd_float, 3),
            self.contact_pairs.contact_info.G[i_p],
            sap_state,
            sap_info,
        )

    @qd.func
    def compute_delassus(
        self,
        i_p: int,
        fem_state: FEMState,
        dofs_state: DofsState,
        links_state: LinksState,
        fem_info: FEMInfo,
        links_info: LinksInfo,
        rigid_info: RigidInfo,
        sap_config: qd.template(),
    ):
        # With floor, J is Identity
        i_v = self.contact_pairs[i_p].geom_idx
        return self.fem_delassus_vertex(i_v, self.contact_pairs[i_p].batch_idx, 1.0, fem_state, fem_info)


@qd.data_oriented
class RigidFloorVertContactHandler(RigidContactHandler):
    def __init__(self, simulator: "Simulator", options: SAPCouplerOptions) -> None:
        super().__init__(simulator)
        rigid_solver = simulator.rigid_solver
        self.name = "RigidFloorVertContactHandler"
        # The key of a pair is its vertex
        self.key_bits = max(1, rigid_solver.n_verts - 1).bit_length()
        self.contact_pair_type = qd.types.struct(
            batch_idx=gs.qd_int,  # batch index
            key=qd.u64,  # key unique among the pairs of the environment (see kernel_order_constraints)
            link_idx=gs.qd_int,  # index of the link
            contact_pos=gs.qd_vec3,  # contact position
            contact_info=self.contact_info_type,
        )
        # Every vertex of a geom lying flat may touch the floor
        n_contacts, _ = get_contact_capacities(options, rigid_solver.n_free_verts, 0)
        self.max_contact_pairs = n_contacts * simulator._B
        self.contact_pairs = self.contact_pair_type.field(shape=(max(self.max_contact_pairs, 1),))

    @qd.func
    def detection(
        self,
        fem_state: FEMState,
        free_verts_state: VertsState,
        fixed_verts_state: VertsState,
        geoms_state: GeomsState,
        sap_state: SAPState,
        contact_queries_state: SAPContactQueriesState,
        fem_info: FEMInfo,
        links_info: LinksInfo,
        verts_info: VertsInfo,
        faces_info: FacesInfo,
        geoms_info: GeomsInfo,
        sap_info: SAPInfo,
        rigid_config: qd.template(),
        errno: qd.Tensor,
    ):
        n_envs = self.envs_n_contact_pairs.shape[0]
        contact_info = qd.static(self.contact_pairs.contact_info)
        self.n_contact_pairs[None] = 0
        for i_v, i_b in qd.ndrange(verts_info.is_fixed.shape[0], n_envs, axes=qd.static(eoa(free_verts_state.pos))):
            if verts_info.is_fixed[i_v]:
                continue
            i_fv = verts_info.verts_state_idx[i_v]
            pos_v = free_verts_state.pos[i_fv, i_b]
            i_g_ground = sap_info.ground_geom_idx[None]
            distance = pos_v.z - geoms_state.pos[i_g_ground, i_b].z
            if distance > 0.0:
                continue
            i_g = verts_info.geom_idx[i_v]
            i_l = geoms_info.link_idx[i_g]
            i_p = func_add_slot(
                i_b,
                self.contact_pairs.shape[0] // n_envs,
                self.envs_n_contact_pairs,
                self.n_contact_pairs,
                code=SAPErrorCode.OVERFLOW_CONTACTS,
                errno=errno,
            )
            if i_p >= 0:
                self.contact_pairs[i_p].batch_idx = i_b
                self.contact_pairs[i_p].key = qd.u64(i_v)
                self.contact_pairs[i_p].link_idx = i_l
                self.contact_pairs[i_p].contact_pos = pos_v
                contact_info[i_p].k = sap_info.rigid_floor_vert_stiffness[None]
                contact_info[i_p].phi0 = distance
                contact_info[i_p].mu = geoms_info.coup_friction[i_g]


@qd.data_oriented
class RigidFloorTetContactHandler(RigidContactHandler):
    has_candidates = True

    def __init__(
        self, simulator: "Simulator", options: SAPCouplerOptions, n_rigid_volume_elems: int, n_floor_volume_elems: int
    ) -> None:
        super().__init__(simulator)
        self.name = "RigidFloorTetContactHandler"
        # The key of a pair is its tet
        self.key_bits = max(1, n_rigid_volume_elems - 1).bit_length()
        self.contact_candidate_type = qd.types.struct(
            batch_idx=gs.qd_int,  # batch index
            geom_idx=gs.qd_int,  # index of the element
            intersection_code=gs.qd_int,  # intersection code for the element
            distance=gs.qd_vec4,  # distance vector for the element
        )
        self.n_contact_candidates = qd.field(gs.qd_int, shape=())
        self.envs_n_contact_candidates = qd.field(gs.qd_int, shape=(simulator._B,))
        # A tet of a link that is not fixed crosses the floor once at most, each crossing giving one pair at most
        n_contacts, n_candidates = get_contact_capacities(options, n_floor_volume_elems, n_floor_volume_elems)
        self.max_contact_candidates = n_candidates * simulator._B
        self.contact_candidates = self.contact_candidate_type.field(shape=(max(self.max_contact_candidates, 1),))

        self.contact_pair_type = qd.types.struct(
            batch_idx=gs.qd_int,  # batch index
            key=qd.u64,  # key unique among the pairs of the environment (see kernel_order_constraints)
            link_idx=gs.qd_int,  # index of the link
            contact_pos=gs.qd_vec3,  # contact position
            contact_info=self.contact_info_type,
        )
        self.max_contact_pairs = n_contacts * simulator._B
        self.contact_pairs = self.contact_pair_type.field(shape=(max(self.max_contact_pairs, 1),))

    @qd.func
    def detection(
        self,
        fem_state: FEMState,
        free_verts_state: VertsState,
        fixed_verts_state: VertsState,
        geoms_state: GeomsState,
        sap_state: SAPState,
        contact_queries_state: SAPContactQueriesState,
        fem_info: FEMInfo,
        links_info: LinksInfo,
        verts_info: VertsInfo,
        faces_info: FacesInfo,
        geoms_info: GeomsInfo,
        sap_info: SAPInfo,
        rigid_config: qd.template(),
        errno: qd.Tensor,
    ):
        n_envs = self.envs_n_contact_candidates.shape[0]
        candidates = qd.static(self.contact_candidates)
        self.n_contact_candidates[None] = 0
        # TODO Check surface element only instead of all elements
        for i_e, i_b in qd.ndrange(
            sap_info.rigid_volume_elems.shape[0], n_envs, axes=qd.static(eoa(sap_state.rigid_volume_verts))
        ):
            i_g = sap_info.rigid_volume_elems_geom_idx[i_e]
            i_l = geoms_info.link_idx[i_g]
            I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l
            if links_info.is_fixed[I_l]:
                continue
            i_g_ground = sap_info.ground_geom_idx[None]
            ground_pos = geoms_state.pos[i_g_ground, i_b]
            intersection_code = qd.int32(0)
            distance = qd.Vector.zero(gs.qd_float, 4)
            for i in qd.static(range(4)):
                i_v = sap_info.rigid_volume_elems[i_e][i]
                distance[i] = sap_state.rigid_volume_verts[i_v, i_b].z - ground_pos.z
                if distance[i] > 0.0:
                    intersection_code |= 1 << i

            # check if the element intersect with the floor
            if intersection_code != 0 and intersection_code != 15:
                i_c = func_add_slot(
                    i_b,
                    candidates.shape[0] // n_envs,
                    self.envs_n_contact_candidates,
                    self.n_contact_candidates,
                    code=SAPErrorCode.OVERFLOW_CANDIDATE_CONTACTS,
                    errno=errno,
                )
                if i_c >= 0:
                    candidates[i_c].batch_idx = i_b
                    candidates[i_c].geom_idx = i_e
                    candidates[i_c].intersection_code = intersection_code
                    candidates[i_c].distance = distance

        pairs = qd.static(self.contact_pairs)
        contact_info = qd.static(pairs.contact_info)
        self.n_contact_pairs[None] = 0
        for i_c in range(self.n_contact_candidates[None]):
            candidate = candidates[i_c]
            i_b = candidate.batch_idx
            i_e = candidate.geom_idx
            distance = candidate.distance
            intersected_edges = sap_info.marching_tets_edge_table[candidate.intersection_code]
            tet_vertices = qd.Matrix.zero(gs.qd_float, 3, 4)  # 4 vertices
            tet_pressures = qd.Vector.zero(gs.qd_float, 4)  # pressures at the vertices
            for i in qd.static(range(4)):
                i_v = sap_info.rigid_volume_elems[i_e][i]
                tet_vertices[:, i] = sap_state.rigid_volume_verts[i_v, i_b]
                tet_pressures[i] = sap_info.rigid_pressure_field[i_v]

            polygon_vertices = qd.Matrix.zero(gs.qd_float, 3, 4)  # 3 or 4 vertices
            total_area = gs.qd_float(0.0)
            total_area_weighted_centroid = qd.Vector.zero(gs.qd_float, 3)
            for i in qd.static(range(4)):
                if intersected_edges[i] >= 0:
                    edge = sap_info.tet_edges[intersected_edges[i]]
                    pos_v0 = tet_vertices[:, edge[0]]
                    pos_v1 = tet_vertices[:, edge[1]]
                    t = distance[edge[0]] / (distance[edge[0]] - distance[edge[1]])
                    polygon_vertices[:, i] = pos_v0 + t * (pos_v1 - pos_v0)

                    # Compute triangle area and centroid
                    if qd.static(i >= 2):
                        accumulate_area_centroid(polygon_vertices, i, total_area, total_area_weighted_centroid)
            if total_area < sap_info.contact_area_min[None]:
                continue
            centroid = total_area_weighted_centroid / total_area

            # Compute barycentric coordinates
            barycentric = tet_barycentric(centroid, tet_vertices)
            pressure = barycentric.dot(tet_pressures)

            grad = sap_state.rigid_pressure_gradient[i_e, i_b]
            g = grad.z
            if g <= sap_info.contact_gradient_rtol[None] * grad.norm():
                continue
            rigid_k = total_area * g
            rigid_phi0 = -pressure / g
            if rigid_k < sap_info.contact_stiffness_min[None] or rigid_phi0 > sap_info.contact_separation_max[None]:
                continue
            i_p = func_add_slot(
                i_b,
                pairs.shape[0] // n_envs,
                self.envs_n_contact_pairs,
                self.n_contact_pairs,
                code=SAPErrorCode.OVERFLOW_CONTACTS,
                errno=errno,
            )
            if i_p >= 0:
                i_g = sap_info.rigid_volume_elems_geom_idx[i_e]
                pairs[i_p].batch_idx = i_b
                pairs[i_p].key = qd.u64(i_e)
                pairs[i_p].link_idx = geoms_info.link_idx[i_g]
                pairs[i_p].contact_pos = centroid
                contact_info[i_p].k = rigid_k
                contact_info[i_p].phi0 = rigid_phi0
                contact_info[i_p].mu = geoms_info.coup_friction[i_g]


@qd.data_oriented
class RigidFemTriTetContactHandler(RigidContactHandler):
    """
    Class for handling contact between rigid triangles and FEM tets in a simulation using hydroelastic model.

    The velocity of a contact is the one of the FEM side relative to the rigid side.
    """

    has_candidates = True
    n_fem_corners = 4

    def __init__(self, simulator: "Simulator", options: SAPCouplerOptions) -> None:
        super().__init__(simulator)
        fem_solver = simulator.fem_solver
        rigid_solver = simulator.rigid_solver
        self.name = "RigidFemTriTetContactHandler"
        # The key of a pair is its rigid face and its FEM tet
        self.element_bits = max(1, fem_solver.n_elements - 1).bit_length()
        self.key_bits = max(1, rigid_solver.n_faces - 1).bit_length() + self.element_bits
        self.contact_candidate_type = qd.types.struct(
            batch_idx=gs.qd_int,  # batch index
            geom_idx0=gs.qd_int,  # index of the FEM element
            geom_idx1=gs.qd_int,  # index of the Rigid Triangle
            vert_idx1=gs.qd_ivec3,  # vertex indices of the rigid triangle
            normal=gs.qd_vec3,  # contact plane normal
            x=gs.qd_vec3,  # a point on the contact plane
        )
        self.n_contact_candidates = qd.field(gs.qd_int, shape=())
        self.envs_n_contact_candidates = qd.field(gs.qd_int, shape=(simulator._B,))
        # Each pair of a triangle of a geom and a surface tet gives one candidate and one contact at most
        n_pairs_max = rigid_solver.n_faces * fem_solver.n_surface_elements
        n_contacts, n_candidates = get_contact_capacities(options, n_pairs_max, n_pairs_max)
        self.max_contact_candidates = n_candidates * simulator._B
        self.contact_candidates = self.contact_candidate_type.field(shape=(max(self.max_contact_candidates, 1),))
        self.contact_pair_type = qd.types.struct(
            batch_idx=gs.qd_int,  # batch index
            key=qd.u64,  # key unique among the pairs of the environment (see kernel_order_constraints)
            normal=gs.qd_vec3,  # contact plane normal
            tangent0=gs.qd_vec3,  # contact plane tangent0
            tangent1=gs.qd_vec3,  # contact plane tangent1
            geom_idx0=gs.qd_int,  # index of the FEM element
            barycentric0=gs.qd_vec4,  # barycentric coordinates of the contact point in tet
            link_idx=gs.qd_int,  # index of the link
            contact_pos=gs.qd_vec3,  # contact position
            contact_info=self.contact_info_type,
        )
        self.max_contact_pairs = n_contacts * simulator._B
        self.contact_pairs = self.contact_pair_type.field(shape=(max(self.max_contact_pairs, 1),))

    @qd.func
    def rigid_side_sign(self, i_side: int):
        # The velocity of a contact is the one of the FEM side relative to the rigid side
        return -1.0

    @qd.func
    def contact_frame(self, i_c: int):
        return qd.Matrix.cols(
            [self.contact_pairs[i_c].tangent0, self.contact_pairs[i_c].tangent1, self.contact_pairs[i_c].normal]
        )

    @qd.func
    def detection(
        self,
        fem_state: FEMState,
        free_verts_state: VertsState,
        fixed_verts_state: VertsState,
        geoms_state: GeomsState,
        sap_state: SAPState,
        contact_queries_state: SAPContactQueriesState,
        fem_info: FEMInfo,
        links_info: LinksInfo,
        verts_info: VertsInfo,
        faces_info: FacesInfo,
        geoms_info: GeomsInfo,
        sap_info: SAPInfo,
        rigid_config: qd.template(),
        errno: qd.Tensor,
    ):
        n_envs = self.envs_n_contact_candidates.shape[0]
        contact_queries_state.rigid_tri.results.count[0] = 0
        func_bvh_query_leaves(contact_queries_state.rigid_tri, func_no_filter, filter_ctx=0)
        func_flag_query_overflow(contact_queries_state.rigid_tri.results, errno)

        # The candidates, the pairs of a rigid triangle and a FEM surface tet crossing its plane
        self.n_contact_candidates[None] = 0
        for i_r in range(contact_queries_state.rigid_tri.results.count[0]):
            i_b, i_t, i_sq = contact_queries_state.rigid_tri.results.triplets[i_r]
            i_a = sap_info.rigid_tri_faces_idx[i_t]
            i_q = fem_info.surface_elements[i_sq]

            vert_idx1 = qd.Vector.zero(gs.qd_int, 3)
            tri_vertices = qd.Matrix.zero(gs.qd_float, 3, 3)
            for i in qd.static(range(3)):
                i_v = faces_info.verts_idx[i_a][i]
                i_fv = verts_info.verts_state_idx[i_v]
                if verts_info.is_fixed[i_v]:
                    tri_vertices[:, i] = fixed_verts_state.pos[i_fv]
                else:
                    tri_vertices[:, i] = free_verts_state.pos[i_fv, i_b]
                vert_idx1[i] = i_v
            pos_v0, pos_v1, pos_v2 = tri_vertices[:, 0], tri_vertices[:, 1], tri_vertices[:, 2]

            e1 = pos_v1 - pos_v0
            e2 = pos_v2 - pos_v0
            normal = e1.cross(e2)
            magnitude = normal.norm()
            # A triangle is degenerate when the sine of its angle at its first vertex is round-off
            if magnitude <= sap_info.precision[None] * e1.norm() * e2.norm():
                continue
            normal = normal / magnitude
            g0 = sap_state.fem_pressure_gradient[i_q, i_b]
            if g0.dot(normal) <= sap_info.contact_gradient_rtol[None] * g0.norm():
                continue

            intersection_code = qd.int32(0)
            for i in qd.static(range(4)):
                i_v = fem_info.elements.el2v[i_q][i]
                if (fem_state.vertices.pos[i_v, i_b] - pos_v0).dot(normal) > 0.0:
                    intersection_code |= 1 << i
            if intersection_code == 0 or intersection_code == 15:
                continue

            i_c = func_add_slot(
                i_b,
                self.contact_candidates.shape[0] // n_envs,
                self.envs_n_contact_candidates,
                self.n_contact_candidates,
                code=SAPErrorCode.OVERFLOW_CANDIDATE_CONTACTS,
                errno=errno,
            )
            if i_c >= 0:
                self.contact_candidates[i_c].batch_idx = i_b
                self.contact_candidates[i_c].normal = normal
                self.contact_candidates[i_c].x = pos_v0
                self.contact_candidates[i_c].geom_idx0 = i_q
                self.contact_candidates[i_c].geom_idx1 = i_a
                self.contact_candidates[i_c].vert_idx1 = vert_idx1

        # The pairs, the triangle clipped by the tet. Intersection code reference:
        # https://github.com/RobotLocomotion/drake/blob/49ab120ec6f5981484918daa821fc7101e10ebc6/geometry/proximity/mesh_intersection.cc
        contact_info = qd.static(self.contact_pairs.contact_info)
        self.n_contact_pairs[None] = 0
        for i_c in range(self.n_contact_candidates[None]):
            i_b = self.contact_candidates[i_c].batch_idx
            i_e = self.contact_candidates[i_c].geom_idx0

            tet_vertices = qd.Matrix.zero(gs.qd_float, 3, 4)  # 4 vertices of tet 0
            tet_pressures = qd.Vector.zero(gs.qd_float, 4)  # pressures at the vertices of tet 0
            polygon_vertices = qd.Matrix.zero(gs.qd_float, 3, 7)  # maximum 7 vertices
            for i in qd.static(range(3)):
                i_v = self.contact_candidates[i_c].vert_idx1[i]
                i_fv = verts_info.verts_state_idx[i_v]
                if verts_info.is_fixed[i_v]:
                    polygon_vertices[:, i] = fixed_verts_state.pos[i_fv]
                else:
                    polygon_vertices[:, i] = free_verts_state.pos[i_fv, i_b]
            for i in qd.static(range(4)):
                i_v = fem_info.elements.el2v[i_e][i]
                tet_vertices[:, i] = fem_state.vertices.pos[i_v, i_b]
                tet_pressures[i] = sap_info.fem_pressure[i_v]
            polygon_n_vertices = gs.qd_int(3)
            if not func_clip_polygon_by_tet(polygon_vertices, polygon_n_vertices, tet_vertices):
                continue

            total_area = gs.qd_float(0.0)
            total_area_weighted_centroid = qd.Vector.zero(gs.qd_float, 3)
            for i in range(2, polygon_n_vertices):
                accumulate_area_centroid(polygon_vertices, i, total_area, total_area_weighted_centroid)
            if total_area < sap_info.contact_area_min[None]:
                continue
            centroid = total_area_weighted_centroid / total_area
            barycentric0 = tet_barycentric(centroid, tet_vertices)
            normal = self.contact_candidates[i_c].normal
            tangent0 = (polygon_vertices[:, 0] - centroid).normalized()
            tangent1 = normal.cross(tangent0)
            deformable_g = sap_info.hydroelastic_stiffness[None]
            grad = sap_state.fem_pressure_gradient[i_e, i_b]
            rigid_g = grad.dot(normal)
            pressure = barycentric0.dot(tet_pressures)
            if rigid_g <= sap_info.contact_gradient_rtol[None] * grad.norm():
                continue
            # TODO: The published hydroelastic model treats the rigid side as infinitely stiff (g = FEM gradient alone).
            # Also, 'hydroelastic_stiffness' is a pressure gradient here but a modulus in the rigid pressure fields.
            g = rigid_g * deformable_g / (deformable_g + rigid_g)  # harmonic average
            i_g = verts_info.geom_idx[self.contact_candidates[i_c].vert_idx1[0]]
            i_p = func_add_slot(
                i_b,
                self.contact_pairs.shape[0] // n_envs,
                self.envs_n_contact_pairs,
                self.n_contact_pairs,
                code=SAPErrorCode.OVERFLOW_CONTACTS,
                errno=errno,
            )
            if i_p >= 0:
                self.contact_pairs[i_p].batch_idx = i_b
                self.contact_pairs[i_p].normal = normal
                self.contact_pairs[i_p].tangent0 = tangent0
                self.contact_pairs[i_p].tangent1 = tangent1
                self.contact_pairs[i_p].key = (
                    qd.u64(self.contact_candidates[i_c].geom_idx1) << qd.u64(self.element_bits)
                ) | qd.u64(i_e)
                self.contact_pairs[i_p].geom_idx0 = i_e
                self.contact_pairs[i_p].barycentric0 = barycentric0
                self.contact_pairs[i_p].link_idx = geoms_info.link_idx[i_g]
                self.contact_pairs[i_p].contact_pos = centroid
                contact_info[i_p].k = total_area * g
                contact_info[i_p].phi0 = -pressure / g
                contact_info[i_p].mu = qd.sqrt(fem_info.elements.friction_mu[i_e] * geoms_info.coup_friction[i_g])

    @qd.func
    def fem_corner_vertex(self, i_c: int, i_corner: int, fem_info: FEMInfo):
        """Return the FEM vertex at a corner of the tet of a contact pair."""
        i_e = self.contact_pairs[i_c].geom_idx0
        return fem_info.elements.el2v[i_e][i_corner]

    @qd.func
    def fem_corner_weight(self, i_c: int, i_corner: int):
        """Return the weight of the velocity of a corner in the velocity of its contact pair."""
        return self.contact_pairs[i_c].barycentric0[i_corner]

    @qd.func
    def compute_delassus(
        self,
        i_p: int,
        fem_state: FEMState,
        dofs_state: DofsState,
        links_state: LinksState,
        fem_info: FEMInfo,
        links_info: LinksInfo,
        rigid_info: RigidInfo,
        sap_config: qd.template(),
    ):
        pairs = qd.static(self.contact_pairs)
        i_b = pairs[i_p].batch_idx
        W = func_link_point_delassus(
            pairs[i_p].link_idx,
            -1,
            i_b,
            pairs[i_p].contact_pos,
            dofs_state,
            links_state,
            links_info,
            rigid_info,
            sap_config,
        )
        i_e = pairs[i_p].geom_idx0
        for i in qd.static(range(4)):
            i_v = fem_info.elements.el2v[i_e][i]
            W = W + pairs[i_p].barycentric0[i] ** 2 * fem_info.dt2_inv[None] * fem_state.vertices_solve.prec[i_v, i_b]
        world = self.contact_frame(i_p)
        return world.transpose() @ W @ world

    @qd.func
    def compute_Jx(
        self,
        i_p: int,
        x_fem: qd.Tensor,
        x_rigid: qd.Tensor,
        dofs_state: DofsState,
        links_state: LinksState,
        fem_info: FEMInfo,
        links_info: LinksInfo,
        sap_config: qd.template(),
    ):
        pairs = qd.static(self.contact_pairs)
        i_b = pairs[i_p].batch_idx
        i_e = pairs[i_p].geom_idx0
        Jx = qd.Vector.zero(gs.qd_float, 3)
        for i in qd.static(range(4)):
            i_v = fem_info.elements.el2v[i_e][i]
            Jx = Jx + pairs[i_p].barycentric0[i] * x_fem[i_v, i_b]
        Jx = Jx - func_link_point_jacobian_product(
            pairs[i_p].link_idx, i_b, pairs[i_p].contact_pos, x_rigid, dofs_state, links_state, links_info, sap_config
        )
        return self.contact_frame(i_p).transpose() @ Jx

    @qd.func
    def coarse_hessian(self, i_p: int, i_side: int, sap_state: SAPState, fem_info: FEMInfo, sap_info: SAPInfo):
        """Return the FEM entity and the Hessian restricted to its rigid motions of a contact pair."""
        i_e = self.contact_pairs[i_p].geom_idx0
        i_v = fem_info.elements.el2v[i_e][0]
        return func_fem_coarse_contact_hessian(
            i_v,
            self.contact_pairs[i_p].batch_idx,
            self.contact_pairs[i_p].contact_pos,
            self.contact_frame(i_p),
            self.contact_pairs.contact_info.G[i_p],
            sap_state,
            sap_info,
        )


@qd.data_oriented
class RigidRigidTetContactHandler(RigidRigidContactHandler):
    """
    Class for handling contact between Rigid bodies using hydroelastic model.

    This class extends the RigidContact class and provides methods for detecting contact
    between tetrahedral elements, computing contact pairs, and managing contact-related computations.
    """

    has_candidates = True

    def __init__(
        self,
        simulator: "Simulator",
        options: SAPCouplerOptions,
        n_rigid_volume_elems: int,
        rigid_volume_elems_geom_idx: np.ndarray,
        collision_pair_idx: np.ndarray,
    ) -> None:
        super().__init__(simulator)
        self.name = "RigidRigidTetContactHandler"
        # The key of a pair is its two tets
        self.element_bits = max(1, n_rigid_volume_elems - 1).bit_length()
        self.key_bits = 2 * self.element_bits
        self.contact_candidate_type = qd.types.struct(
            batch_idx=gs.qd_int,  # batch index
            geom_idx0=gs.qd_int,  # index of the element
            geom_idx1=gs.qd_int,  # index of the other element
            intersection_code0=gs.qd_int,  # intersection code for element0
            normal=gs.qd_vec3,  # contact plane normal
            x=gs.qd_vec3,  # a point on the contact plane
            distance0=gs.qd_vec4,  # distance vector for element0
        )
        self.n_contact_candidates = qd.field(gs.qd_int, shape=())
        self.envs_n_contact_candidates = qd.field(gs.qd_int, shape=(simulator._B,))
        # Each pair of tets of two geoms that may collide gives one candidate and one contact at most
        geoms_n_tets = np.bincount(rigid_volume_elems_geom_idx, minlength=simulator.rigid_solver.n_geoms)
        is_pair = np.triu((collision_pair_idx >= 0) | (collision_pair_idx.T >= 0), k=1)
        n_pairs_max = int((np.outer(geoms_n_tets, geoms_n_tets) * is_pair).sum())
        n_contacts, n_candidates = get_contact_capacities(options, n_pairs_max, n_pairs_max)
        self.max_contact_candidates = n_candidates * simulator._B
        self.contact_candidates = self.contact_candidate_type.field(shape=(max(self.max_contact_candidates, 1),))

        self.contact_pair_type = qd.types.struct(
            batch_idx=gs.qd_int,  # batch index
            key=qd.u64,  # key unique among the pairs of the environment (see kernel_order_constraints)
            normal=gs.qd_vec3,  # contact plane normal
            tangent0=gs.qd_vec3,  # contact plane tangent0
            tangent1=gs.qd_vec3,  # contact plane tangent1
            link_idx0=gs.qd_int,  # index of the link
            link_idx1=gs.qd_int,  # index of the other link
            contact_pos=gs.qd_vec3,  # contact position
            contact_info=self.contact_info_type,
        )
        self.max_contact_pairs = n_contacts * simulator._B
        self.contact_pairs = self.contact_pair_type.field(shape=(max(self.max_contact_pairs, 1),))

    @qd.func
    def detection(
        self,
        fem_state: FEMState,
        free_verts_state: VertsState,
        fixed_verts_state: VertsState,
        geoms_state: GeomsState,
        sap_state: SAPState,
        contact_queries_state: SAPContactQueriesState,
        fem_info: FEMInfo,
        links_info: LinksInfo,
        verts_info: VertsInfo,
        faces_info: FacesInfo,
        geoms_info: GeomsInfo,
        sap_info: SAPInfo,
        rigid_config: qd.template(),
        errno: qd.Tensor,
    ):
        n_envs = self.envs_n_contact_candidates.shape[0]
        contact_queries_state.rigid_tet.results.count[0] = 0
        func_bvh_query_leaves(
            contact_queries_state.rigid_tet,
            func_filter_rigid_tets,
            qd.static((sap_info.rigid_volume_elems_geom_idx, sap_info.rigid_collision_pair_idx)),
        )
        func_flag_query_overflow(contact_queries_state.rigid_tet.results, errno)

        # The candidates, the pairs of tets crossing their equal-pressure plane
        candidates = qd.static(self.contact_candidates)
        self.n_contact_candidates[None] = 0
        for i_r in range(contact_queries_state.rigid_tet.results.count[0]):
            i_b, i_a, i_q = contact_queries_state.rigid_tet.results.triplets[i_r]
            i_v0 = sap_info.rigid_volume_elems[i_a][0]
            i_v1 = sap_info.rigid_volume_elems[i_q][1]
            x0 = sap_state.rigid_volume_verts[i_v0, i_b]
            x1 = sap_state.rigid_volume_verts[i_v1, i_b]
            p0 = sap_info.rigid_pressure_field[i_v0]
            p1 = sap_info.rigid_pressure_field[i_v1]
            g0 = sap_state.rigid_pressure_gradient[i_a, i_b]
            g1 = sap_state.rigid_pressure_gradient[i_q, i_b]
            g0_norm = g0.norm()
            g1_norm = g1.norm()
            if g0_norm == 0.0 or g1_norm == 0.0:
                continue
            # Calculate the isosurface, i.e. equal pressure plane defined by x and normal
            # Solve for p0 + g0.dot(x - x0) = p1 + g1.dot(x - x1)
            normal = g0 - g1
            magnitude = normal.norm()
            g_rtol = sap_info.contact_gradient_rtol[None]
            if magnitude <= g_rtol * (g0_norm + g1_norm):
                continue
            normal = normal / magnitude
            b = p1 - p0 - g1.dot(x1) + g0.dot(x0)
            x = b / magnitude * normal
            # Check that the normal is pointing along g0 and against g1
            if normal.dot(g0) <= g_rtol * g0_norm or normal.dot(g1) >= -g_rtol * g1_norm:
                continue

            intersection_code0 = qd.int32(0)
            distance0 = qd.Vector.zero(gs.qd_float, 4)
            intersection_code1 = qd.int32(0)
            for i in qd.static(range(4)):
                i_v = sap_info.rigid_volume_elems[i_a][i]
                distance0[i] = (sap_state.rigid_volume_verts[i_v, i_b] - x).dot(normal)  # signed distance
                if distance0[i] > 0.0:
                    intersection_code0 |= 1 << i
            for i in qd.static(range(4)):
                i_v = sap_info.rigid_volume_elems[i_q][i]
                if (sap_state.rigid_volume_verts[i_v, i_b] - x).dot(normal) > 0.0:
                    intersection_code1 |= 1 << i
            # Fast check for whether both tets intersect with the plane
            if (
                intersection_code0 == 0
                or intersection_code1 == 0
                or intersection_code0 == 15
                or intersection_code1 == 15
            ):
                continue
            i_c = func_add_slot(
                i_b,
                candidates.shape[0] // n_envs,
                self.envs_n_contact_candidates,
                self.n_contact_candidates,
                code=SAPErrorCode.OVERFLOW_CANDIDATE_CONTACTS,
                errno=errno,
            )
            if i_c >= 0:
                candidates[i_c].batch_idx = i_b
                candidates[i_c].normal = normal
                candidates[i_c].x = x
                candidates[i_c].geom_idx0 = i_a
                candidates[i_c].intersection_code0 = intersection_code0
                candidates[i_c].distance0 = distance0
                candidates[i_c].geom_idx1 = i_q

        # The pairs, the polygon of the plane in tet 0 clipped by tet 1
        pairs = qd.static(self.contact_pairs)
        contact_info = qd.static(pairs.contact_info)
        self.n_contact_pairs[None] = 0
        for i_c in range(self.n_contact_candidates[None]):
            i_b = candidates[i_c].batch_idx
            i_e0 = candidates[i_c].geom_idx0
            i_e1 = candidates[i_c].geom_idx1
            distance0 = candidates[i_c].distance0
            intersected_edges0 = sap_info.marching_tets_edge_table[candidates[i_c].intersection_code0]
            tet_vertices0 = qd.Matrix.zero(gs.qd_float, 3, 4)  # 4 vertices of tet 0
            tet_pressures0 = qd.Vector.zero(gs.qd_float, 4)  # pressures at the vertices of tet 0
            tet_vertices1 = qd.Matrix.zero(gs.qd_float, 3, 4)  # 4 vertices of tet 1
            for i in qd.static(range(4)):
                i_v = sap_info.rigid_volume_elems[i_e0][i]
                tet_vertices0[:, i] = sap_state.rigid_volume_verts[i_v, i_b]
                tet_pressures0[i] = sap_info.rigid_pressure_field[i_v]
            for i in qd.static(range(4)):
                i_v = sap_info.rigid_volume_elems[i_e1][i]
                tet_vertices1[:, i] = sap_state.rigid_volume_verts[i_v, i_b]

            polygon_vertices = qd.Matrix.zero(gs.qd_float, 3, 8)  # maximum 8 vertices
            polygon_n_vertices = gs.qd_int(0)
            for i in range(4):
                if intersected_edges0[i] >= 0:
                    edge = sap_info.tet_edges[intersected_edges0[i]]
                    pos_v0 = tet_vertices0[:, edge[0]]
                    pos_v1 = tet_vertices0[:, edge[1]]
                    t = distance0[edge[0]] / (distance0[edge[0]] - distance0[edge[1]])
                    polygon_vertices[:, polygon_n_vertices] = pos_v0 + t * (pos_v1 - pos_v0)
                    polygon_n_vertices = polygon_n_vertices + 1
            if not func_clip_polygon_by_tet(polygon_vertices, polygon_n_vertices, tet_vertices1):
                continue

            # compute centroid and area of the polygon
            total_area = gs.qd_float(0.0)
            total_area_weighted_centroid = qd.Vector.zero(gs.qd_float, 3)
            for i in range(2, polygon_n_vertices):
                accumulate_area_centroid(polygon_vertices, i, total_area, total_area_weighted_centroid)
            if total_area < sap_info.contact_area_min[None]:
                continue
            centroid = total_area_weighted_centroid / total_area
            normal = candidates[i_c].normal
            tangent0 = polygon_vertices[:, 0] - centroid
            tangent0 = tangent0 / tangent0.norm()
            tangent1 = normal.cross(tangent0)
            grad0 = sap_state.rigid_pressure_gradient[i_e0, i_b]
            grad1 = sap_state.rigid_pressure_gradient[i_e1, i_b]
            g0 = grad0.dot(normal)
            g1 = -grad1.dot(normal)
            g_rtol = sap_info.contact_gradient_rtol[None]
            if g0 <= g_rtol * grad0.norm() or g1 <= g_rtol * grad1.norm():
                continue
            g = 1.0 / (1.0 / g0 + 1.0 / g1)  # harmonic average
            barycentric0 = tet_barycentric(centroid, tet_vertices0)
            rigid_phi0 = -barycentric0.dot(tet_pressures0) / g
            if rigid_phi0 > sap_info.contact_separation_max[None]:
                continue
            i_p = func_add_slot(
                i_b,
                pairs.shape[0] // n_envs,
                self.envs_n_contact_pairs,
                self.n_contact_pairs,
                code=SAPErrorCode.OVERFLOW_CONTACTS,
                errno=errno,
            )
            if i_p >= 0:
                i_g0 = sap_info.rigid_volume_elems_geom_idx[i_e0]
                i_g1 = sap_info.rigid_volume_elems_geom_idx[i_e1]
                pairs[i_p].batch_idx = i_b
                pairs[i_p].key = (qd.u64(i_e0) << qd.u64(self.element_bits)) | qd.u64(i_e1)
                pairs[i_p].normal = normal
                pairs[i_p].tangent0 = tangent0
                pairs[i_p].tangent1 = tangent1
                pairs[i_p].contact_pos = centroid
                pairs[i_p].link_idx0 = geoms_info.link_idx[i_g0]
                pairs[i_p].link_idx1 = geoms_info.link_idx[i_g1]
                contact_info[i_p].k = total_area * g
                contact_info[i_p].phi0 = rigid_phi0
                contact_info[i_p].mu = qd.sqrt(geoms_info.friction[i_g0] * geoms_info.friction[i_g1])
