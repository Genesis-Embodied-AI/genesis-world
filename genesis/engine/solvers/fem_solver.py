from collections.abc import Iterator, Sequence
from enum import IntEnum, auto
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import torch

import igl

import quadrants as qd

import genesis as gs
import genesis.utils.simt as su
from genesis.engine.entities.fem_entity import FEMEntity
from genesis.engine.materials import FEM
from genesis.engine.materials.FEM.elastic import (
    FEMModel,
    func_linear_corotated_energy,
    func_linear_corotated_energy_gradient,
    func_linear_corotated_hessian_product,
    func_linear_corotated_rotation,
    func_linear_corotated_rotation_grad,
    func_linear_energy,
    func_linear_energy_gradient,
    func_linear_hessian_product,
    func_linear_stress,
    func_stable_neohookean_energy,
    func_stable_neohookean_stress,
)
from genesis.engine.materials.FEM.muscle import (
    func_muscle_fiber_dir,
    func_muscle_fiber_dir_grad,
    func_muscle_fiber_stress,
    func_muscle_stress,
)
from genesis.engine.solvers.rigid.data import LinksState
from genesis.engine.state_change import StateChange, mutates
from genesis.engine.states.solvers import SolverCheckpoint
from genesis.grad.bridge import custom_backward, no_grad
from genesis.options.solvers import FEMOptions
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
)
from genesis.utils.geom import qd_inv_quat, qd_transform_by_quat, qd_transform_quat_by_quat
from genesis.utils.misc import (
    broadcast_tensor,
    get_gpu_core_count,
    get_gpu_cores_per_unit,
    qd_isfinite,
    qd_to_numpy,
    qd_to_torch,
    qd_zero_grad,
    tensor_to_array,
)
from genesis.utils.warnings import warn_once

from .base_solver import (
    VFACES,
    VVERTS,
    ErrorCode,
    GravityMixin,
    Solver,
    TimeBasedMixin,
    kernel_get_values,
    kernel_set_actu,
)

if TYPE_CHECKING:
    from genesis.engine.scene import Scene
    from genesis.engine.simulator import Simulator

# The most clusters an entity splits into, which bounds the size of its coarse matrix, the fewest vertices a cluster
# holds, and the most clusters of a scene beyond one per entity (see 'FEMSolver.build'). The coarse matrices of an
# environment then hold at most 36 * MAX_ENTITY_CLUSTERS * MAX_SCENE_CLUSTERS single precision entries, 18 KiB, a small
# share of the memory of an environment whatever the number of entities.
MAX_ENTITY_CLUSTERS = 8
MIN_CLUSTER_VERTICES = 64
MAX_SCENE_CLUSTERS = 16

# Default threshold of the PCG solves of a forward simulation on their squared residual relative to its initial one, by
# precision (see FEMOptions)
PCG_THRESHOLD_DEFAULT = {np.float64: 1e-9, np.float32: 1e-6}


class FEMErrorCode(ErrorCode):
    """The error flags the finite element method (FEM) solver writes in its errno (see 'Solver.register_errors')."""

    INVALID_VERTEX_NAN = auto()


class FEMSolveStage(IntEnum):
    """The stages of the implicit solve that the host launches one by one (see 'FEMSolver._solve_on_host')."""

    INIT_NEWTON = auto()
    NEWTON_HEAD = auto()
    PCG_ITER = auto()
    INIT_LINESEARCH = auto()
    LINESEARCH_ITER = auto()
    SKIP_LINESEARCH = auto()
    END_NEWTON = auto()
    INIT_ADJOINT = auto()
    SEED_ADJOINT = auto()


VERTICES = Size("vertices")
ELEMENTS = Size("elements")
SURFACES = Size("surfaces")
SURFACE_VERTICES = Size("surface_vertices")
SURFACE_ELEMENTS = Size("surface_elements")
# The clusters of vertices whose rigid motions span the coarse space of the preconditioner of the implicit solve (see
# 'FEMCoarseSolve'), the vertices and the elements they list, the entities they split and the entries of the coarse
# matrices.
CLUSTERS = Size("clusters")
CLUSTERS_VERTICES = Size("clusters_vertices")
CLUSTERS_ELEMENTS = Size("clusters_elements")
COARSE_ENTITIES = Size("coarse_entities")
COARSE_ENTRIES = Size("coarse_entries")
# The slots of the vertex constraints of an environment (see 'FEMOptions.max_vertex_constraints')
CONSTRAINT_SLOTS = Size("constraint_slots")

# The arrays of the implicit solve, which the explicit solver leaves out
IMPLICIT = "implicit"
# The rotation of the elements of the linear corotated model, and the fiber directions of the muscles under the
# implicit solver.
COROTATED = "corotated"
MUSCLE_FIBERS = "muscle_fibers"
# The corners of the elements gathered by vertex, which the GPU kernels walk (see 'vertices_corners_idx')
CORNER_BLOCKS = "corner_blocks"


class FEMVertices(Data, kind=DataKind.STATE, shape=(VERTICES, ENVS)):
    """
    Runtime state of the finite element method (FEM) vertices, one value per vertex and environment.

    A substep writes the positions and velocities it integrates in `pos_next` and `vel_next`, which it copies to the
    state once over (see 'kernel_copy_next_to_curr').
    """

    pos: qd.Tensor = field((float, 3), grad=True)
    vel: qd.Tensor = field((float, 3), grad=True)
    pos_next: qd.Tensor = field((float, 3), kind=DataKind.SCRATCH, grad=True)
    vel_next: qd.Tensor = field((float, 3), kind=DataKind.SCRATCH, grad=True)


class FEMElements(Data, kind=DataKind.STATE, shape=(ELEMENTS, ENVS)):
    """
    Runtime state of the finite element method (FEM) elements, one value per element and environment.

    It holds their actuation.
    """

    actu: qd.Tensor = field(float, grad=True)


class FEMElementsEnergy(Data, kind=DataKind.SCRATCH, shape=(ELEMENTS, ENVS)):
    """
    The energy density of each element at the end of a substep of the implicit solver, and the stress its vertices
    gather.
    """

    energy: qd.Tensor = field(float)
    # The stress times the rest volume: minus the gradient of the energy density with respect to the deformation
    # gradient plus the Rayleigh stiffness damping where a Newton iteration starts, then on GPU the product of its
    # Hessian with the change of deformation along the search direction of each PCG iteration. Each element computes it
    # once, then each of its corners multiplies it by its row of the mapping matrix (see 'func_element_mapping_matrix').
    stress: qd.Tensor = field((float, (3, 3)))


class FEMElementsMaterial(Data, kind=DataKind.DERIVED, shape=(ELEMENTS, ENVS)):
    """
    The data the materials compute for each element from the deformation at the start of a substep.

    `rotation` is the rotation the linear corotated model linearizes about, and `fiber_dir` the direction of the fibers
    of a muscle, along which the implicit solver applies its active stress.
    """

    rotation: qd.Tensor = field((float, (3, 3)), labels=COROTATED)
    fiber_dir: qd.Tensor = field((float, 3), labels=MUSCLE_FIBERS)


class FEMVerticesEnergy(Data, kind=DataKind.SCRATCH, shape=(VERTICES, ENVS)):
    """
    The inertia of each vertex at the end of a substep of the implicit solver, and the force on it there.

    The adjoint of the forces holds the solution of the adjoint solve (see 'FEMSolver._implicit_solve_grad'), which the
    inertia carries back to the state at the start of the substep.
    """

    inertia: qd.Tensor = field((float, 3), grad=True)
    force: qd.Tensor = field((float, 3), grad=True)


class FEMVerticesSolve(Data, kind=DataKind.SCRATCH, shape=(VERTICES, ENVS)):
    """
    The preconditioned conjugate gradient (PCG) solve and the line search of the implicit solver, for each vertex.

    `diag3x3` is the diagonal 3-by-3 block of the Hessian, `prec` its inverse, the block Jacobi preconditioner, `x`
    the solution, `r` the residual, `z` the preconditioned residual, `p` the search direction, `Ap` the product of the
    Hessian with it, and `x_prev` the position the line search starts from.
    """

    diag3x3: qd.Tensor = field((float, (3, 3)))
    prec: qd.Tensor = field((float, (3, 3)))
    x: qd.Tensor = field((float, 3))
    r: qd.Tensor = field((float, 3))
    z: qd.Tensor = field((float, 3))
    p: qd.Tensor = field((float, 3))
    Ap: qd.Tensor = field((float, 3))
    x_prev: qd.Tensor = field((float, 3))


class FEMEnvsSolve(Data, kind=DataKind.SCRATCH, shape=(ENVS,)):
    """
    The progress and scalars of the implicit solve of each environment.

    They are whether its Newton iterations, its preconditioned conjugate gradient (PCG) solve and its line search go on,
    the scalars of the PCG solve, the squared residual below which its PCG solve stops (see 'pcg_threshold' in
    FEMInfo), and the energy, step and slope of the line search.
    """

    is_active: qd.Tensor = field(bool)
    is_pcg_active: qd.Tensor = field(bool)
    is_linesearch_active: qd.Tensor = field(bool)
    rTr: qd.Tensor = field(float)
    rTz: qd.Tensor = field(float)
    rTr_threshold: qd.Tensor = field(float)
    prev_energy: qd.Tensor = field(float)
    step_size: qd.Tensor = field(float)
    m: qd.Tensor = field(float)


class FEMCoarseSolve(Data, kind=DataKind.SCRATCH):
    """
    The coarse space of the preconditioner of the preconditioned conjugate gradient (PCG) solve.

    The block Jacobi preconditioner damps the local errors alone, while the motion of a whole entity and its smooth
    deformations couple all its vertices. The rigid motions of the clusters of each entity span them, each one holding a
    translation, then a rotation about the first vertex of its cluster.
    """

    # The inverse of the Hessian of the substep restricted to the rigid motions of the clusters of each entity, scaled
    # to a unit diagonal, which keeps the rounding errors of the inversion small. Single precision whatever the
    # precision of the simulation, deliberately: the first array of Genesis whose precision does not follow the one the
    # user picks. The matrix preconditions the PCG solve alone, so its precision changes how fast the solve converges,
    # never the solution it converges to, while it is the largest array of the coarse space, quadratic in the clusters
    # of an entity. No Genesis dtype names single precision regardless of the simulation precision, hence qd.f32.
    matrix: qd.Tensor = field(qd.f32, shape=(COARSE_ENTRIES, ENVS))
    # The inverse square root of the diagonal entry of each rigid motion, applied in the precision of the simulation
    # before the matrix narrows to single precision, which keeps its entries in range.
    scale: qd.Tensor = field(float, shape=(6 * CLUSTERS, ENVS))
    # The residual restricted to the rigid motions of each cluster, and the rigid motion of each cluster solving it
    rhs: qd.Tensor = field((float, 6), shape=(CLUSTERS, ENVS))
    sol: qd.Tensor = field((float, 6), shape=(CLUSTERS, ENVS))


class FEMSolveLoops(Data, kind=DataKind.SCRATCH, shape=()):
    """
    The iterations left to each loop of the implicit solve, and the flags that keep the loops running on the device as
    long as an environment iterates on (see 'kernel_batch_solve').

    The flags are ndarrays on every backend, since graph.do_while takes the same ndarray on every call.
    """

    n_newton_left: qd.Tensor = field(int)
    n_pcg_left: qd.Tensor = field(int)
    n_linesearch_left: qd.Tensor = field(int)
    newton_counter: qd.Tensor = field(qd.i32, ndarray=True)
    pcg_counter: qd.Tensor = field(qd.i32, ndarray=True)
    linesearch_counter: qd.Tensor = field(qd.i32, ndarray=True)


class FEMVVertsRender(Data, kind=DataKind.SCRATCH, shape=(VVERTS, ENVS)):
    """The visual vertices as drawn, each one at the position of the simulated vertex it stands for."""

    pos: qd.Tensor = field((float, 3))


class FEMState(Data, kind=DataKind.STATE):
    """Runtime state of the finite element method (FEM) solver."""

    vertices: FEMVertices
    elements: FEMElements
    elements_material: FEMElementsMaterial
    elements_energy: FEMElementsEnergy = field(labels=IMPLICIT)
    vertices_energy: FEMVerticesEnergy = field(labels=IMPLICIT)
    vertices_solve: FEMVerticesSolve = field(labels=IMPLICIT)
    envs_solve: FEMEnvsSolve = field(labels=IMPLICIT)
    coarse_solve: FEMCoarseSolve = field(labels=IMPLICIT)
    solve_loops: FEMSolveLoops = field(labels=IMPLICIT)
    vverts_render: FEMVVertsRender


class FEMElementsInfo(Data, kind=DataKind.CONSTANT, shape=(ELEMENTS,)):
    """
    Model parameters of the FEM elements, one value per element.

    `el2v` holds the vertices of an element, `mu` and `lam` its Lame parameters, `mass_scaled` its mass multiplied by
    the volume scale of the solver, `B` the inverse of its deformation gradient at rest, `V` its rest volume, `model`
    the constitutive model of its material, `is_muscle` whether its material is a muscle, of active stiffness
    `muscle_stiffness`, whose fibers follow `muscle_direction` and its group `muscle_group`.
    """

    el2v: qd.Tensor = field((int, 4))
    mu: qd.Tensor = field(float)
    lam: qd.Tensor = field(float)
    mass_scaled: qd.Tensor = field(float)
    B: qd.Tensor = field((float, (3, 3)))
    V: qd.Tensor = field(float)
    friction_mu: qd.Tensor = field(float)
    model: qd.Tensor = field(int)
    is_muscle: qd.Tensor = field(bool)
    muscle_stiffness: qd.Tensor = field(float)
    muscle_group: qd.Tensor = field(int, kind=DataKind.INFO)
    muscle_direction: qd.Tensor = field((float, 3), kind=DataKind.INFO)


class FEMVerticesInfo(Data, kind=DataKind.CONSTANT, shape=(VERTICES,)):
    """
    Model parameters of the finite element method (FEM) vertices.

    They are their mass, its inverse, its ratio to the squared substep and friction.
    """

    mass: qd.Tensor = field(float)
    mass_inv: qd.Tensor = field(float)
    mass_over_dt2: qd.Tensor = field(float)
    friction_mu: qd.Tensor = field(float)


class FEMSurfacesInfo(Data, kind=DataKind.CONSTANT, shape=(SURFACES,)):
    """
    The surface triangles of the finite element method (FEM) entities, for coupling.

    They hold their vertices and the element each one bounds.
    """

    tri2v: qd.Tensor = field((int, 3))
    tri2el: qd.Tensor = field(int)


class FEMInfo(Data, kind=DataKind.CONSTANT):
    """Model parameters of the finite element method (FEM) solver."""

    elements: FEMElementsInfo
    vertices: FEMVerticesInfo
    surfaces: FEMSurfacesInfo
    # The simulated vertex each visual vertex stands for
    vverts_vert_idx: qd.Tensor = field(int, shape=(VVERTS,))
    vverts_uvs: qd.Tensor = field((float, 2), shape=(VVERTS,))
    vfaces_indices: qd.Tensor = field((int, 3), shape=(VFACES,))
    gravity: qd.Tensor = field((float, 3), shape=(ENVS,), kind=DataKind.INFO)
    substep_dt: qd.Tensor = field(float, shape=())
    dt2: qd.Tensor = field(float, shape=())
    dt2_inv: qd.Tensor = field(float, shape=())
    # The factor by which the explicit solver damps the velocities over a substep
    damping_decay: qd.Tensor = field(float, shape=())
    # The Rayleigh mass damping times the substep and the stiffness damping over it, for the implicit solver
    damping_alpha_dt: qd.Tensor = field(float, shape=())
    damping_beta_over_dt: qd.Tensor = field(float, shape=())
    # One plus 'damping_beta_over_dt', the factor of the elastic Hessian in the Hessian of a substep
    damping_beta_factor: qd.Tensor = field(float, shape=())
    # The factor scaling the masses of the elements, for numerical robustness
    vol_scale: qd.Tensor = field(float, shape=())
    # The largest deviation from a rotation that the polar decomposition of the linear corotated model accepts from its
    # Newton iteration (see 'func_linear_corotated_rotation').
    rotation_atol: qd.Tensor = field(float, shape=())
    # Added to the squared length of a muscle fiber before taking its root
    fiber_length_eps: qd.Tensor = field(float, shape=())
    n_newton_iterations: qd.Tensor = field(int, shape=(), labels=IMPLICIT)
    n_pcg_iterations: qd.Tensor = field(int, shape=(), labels=IMPLICIT)
    n_linesearch_iterations: qd.Tensor = field(int, shape=(), labels=IMPLICIT)
    # The squared residual at which a PCG solve stops, relative to its initial one, at least the square of the precision
    # of the simulation, below which the residual is round-off
    pcg_threshold: qd.Tensor = field(float, shape=(), labels=IMPLICIT)
    # The root mean square of the displacements of the vertices over a Newton iteration below which an environment
    # stops iterating
    newton_dx_threshold: qd.Tensor = field(float, shape=(), labels=IMPLICIT)
    # The sufficient decrease constant and the backtracking factor of the line search
    linesearch_c: qd.Tensor = field(float, shape=(), labels=IMPLICIT)
    linesearch_tau: qd.Tensor = field(float, shape=(), labels=IMPLICIT)
    # The pivot of a coarse matrix scaled to a unit diagonal below which the elimination drops a rigid motion, which the
    # constrained vertices or a cluster too small to rotate leave without inertia.
    coarse_pivot_rtol: qd.Tensor = field(float, shape=(), labels=IMPLICIT)
    # Whether some element was added with its vertices in the wrong order
    has_inverted_elements: qd.Tensor = field(bool, shape=(), kind=DataKind.SCRATCH)
    # The vertices of the surface triangles, and the lumped mass of the surface at each one
    surface_vertices: qd.Tensor = field(int, shape=(SURFACE_VERTICES,))
    surface_vert_mass: qd.Tensor = field(float, shape=(SURFACE_VERTICES,))
    # The triangles incident to each surface vertex in increasing order, in compressed sparse row (CSR) layout, with the
    # scaled mass of the element each one bounds.
    surface_vertices_tri_start: qd.Tensor = field(int, shape=(SURFACE_VERTICES + 1,))
    surface_vertices_tri_idx: qd.Tensor = field(int, shape=(3 * SURFACES,))
    surface_vertices_tri_mass_scaled: qd.Tensor = field(float, shape=(3 * SURFACES,))
    # The elements touching the surface triangles
    surface_elements: qd.Tensor = field(int, shape=(SURFACE_ELEMENTS,))
    # The corners of each vertex in CSR layout, element by element, each one packed as 4 times its element plus its
    # local index. A vertex gathers the terms of its corners in this order, so the implicit solve and the internal force
    # getter sum them without atomics.
    vertices_corners_start: qd.Tensor = field(int, shape=(VERTICES + 1,), labels=CORNER_BLOCKS)
    vertices_corners_idx: qd.Tensor = field(int, shape=(4 * ELEMENTS,), labels=CORNER_BLOCKS)
    # The cluster of each vertex (see 'FEMCoarseSolve'), -1 for a vertex of a cloth, which the solve leaves out
    vertices_cluster_idx: qd.Tensor = field(int, shape=(VERTICES,), labels=IMPLICIT)
    # The entity of elements each cluster belongs to, the clusters of an entity following each other
    clusters_entity_idx: qd.Tensor = field(int, shape=(CLUSTERS,), labels=IMPLICIT)
    # The vertices of each cluster in CSR layout, the first one being the point its rotations are about
    clusters_vertices_start: qd.Tensor = field(int, shape=(CLUSTERS + 1,), labels=IMPLICIT)
    clusters_vertices_idx: qd.Tensor = field(int, shape=(CLUSTERS_VERTICES,), labels=IMPLICIT)
    # The elements touching each cluster in increasing order, in CSR layout
    clusters_elements_start: qd.Tensor = field(int, shape=(CLUSTERS + 1,), labels=IMPLICIT)
    clusters_elements_idx: qd.Tensor = field(int, shape=(CLUSTERS_ELEMENTS,), labels=IMPLICIT)
    # The first cluster of each entity of elements, and the first entry of its coarse matrix, of 6 rows and columns per
    # cluster.
    coarse_entities_cluster_start: qd.Tensor = field(int, shape=(COARSE_ENTITIES + 1,), labels=IMPLICIT)
    coarse_entities_entry_start: qd.Tensor = field(int, shape=(COARSE_ENTITIES + 1,), labels=IMPLICIT)


# TODO: constraints set after a saved state survive 'scene.reset' (see MPMParticleConstraints in mpm_solver.py)
class FEMVertexConstraints(Data, kind=DataKind.INFO, shape=(CONSTRAINT_SLOTS, ENVS)):
    """
    The constraints of the FEM vertices of each environment, in its first slots.

    A constrained vertex is set to `target_pos`, or pulled toward it by a critically damped spring of stiffness
    `stiffness` for a soft constraint, the target following the rigid link `link_idx` when there is one, at the offset
    `link_offset_pos` from where the link stood at `link_init_quat`.
    """

    # Slot of the constraint of each vertex, -1 for a free vertex, and slots in use by environment
    vertices_slot: qd.Tensor = field(int, shape=(VERTICES, ENVS))
    envs_n_constraints: qd.Tensor = field(int, shape=(ENVS,))
    vertex_idx: qd.Tensor = field(int)
    target_pos: qd.Tensor = field((float, 3))
    is_soft_constraint: qd.Tensor = field(bool)
    stiffness: qd.Tensor = field(float)
    link_idx: qd.Tensor = field(int)
    link_offset_pos: qd.Tensor = field((float, 3))
    link_init_quat: qd.Tensor = field((float, 4))


class FEMConfig(Config):
    """The compile-time configuration of the kernels of the finite element method (FEM) solver."""

    use_implicit_solver: bool
    # Whether a block of threads sums the terms of each environment and the lanes of a subgroup tile those of each
    # vertex, on GPU, where a single thread runs each environment on CPU
    enable_cooperative_kernels: bool
    # The bound on the clusters of an entity that sizes the coarse matrices in the kernels, a power of two up to
    # MAX_ENTITY_CLUSTERS (see 'FEMSolver.build')
    max_entity_clusters: int
    # Whether one block of threads runs the whole PCG solve of an environment (see 'FEMSolver.build')
    is_pcg_solved_per_env_block: bool
    has_linesearch: bool
    has_damping_alpha: bool
    has_damping_beta: bool
    # The constitutive models the elements use, the one they all share or -1 (see 'FEMSolver.build'), and whether some
    # element is a muscle.
    has_linear: bool
    has_stable_neohookean: bool
    has_linear_corotated: bool
    shared_model: int
    has_muscle: bool


class FEMClusters(NamedTuple):
    """The clusters of the vertices of the entities of elements, and the elements touching them (see FEMInfo)."""

    vertices_cluster_idx: np.ndarray
    clusters_entity_idx: np.ndarray
    clusters_vertices_start: np.ndarray
    clusters_vertices_idx: np.ndarray
    clusters_elements_start: np.ndarray
    clusters_elements_idx: np.ndarray
    coarse_entities_cluster_start: np.ndarray
    coarse_entities_entry_start: np.ndarray


class FEMData(Data):
    """Every array of the FEM solver but its vertex constraints, allocated from the solver once its scene is built.

    `n_surface_vertices` and `n_surface_elements` are the numbers of vertices and elements on the surface of the
    entities, which the solver derives from their topology, and `clusters` the clusters of their vertices.
    """

    fem_info: FEMInfo
    fem_state: FEMState
    errno: qd.Tensor = field(int, shape=(ENVS,), kind=DataKind.STATE)

    def __init__(self, solver: "FEMSolver", n_surface_vertices: int, n_surface_elements: int, clusters: FEMClusters):
        config = solver.fem_config
        is_implicit = config.use_implicit_solver
        info_values = {}
        unused = {(FEMData.fem_info, FEMData.fem_state): True}
        if solver.is_active:
            substep_dt = solver.substep_dt
            info_values = dict(
                substep_dt=substep_dt,
                dt2=substep_dt**2,
                dt2_inv=1.0 / (substep_dt**2),
                damping_decay=np.exp(-substep_dt * solver._damping),
                damping_alpha_dt=solver._damping_alpha * substep_dt,
                damping_beta_over_dt=solver._damping_beta / substep_dt,
                damping_beta_factor=solver._damping_beta / substep_dt + 1.0,
                vol_scale=solver._vol_scale,
                rotation_atol=1e-10,
                fiber_length_eps=1e-12,
            )
            if is_implicit:
                info_values.update(
                    n_newton_iterations=solver._n_newton_iterations,
                    n_pcg_iterations=solver._n_pcg_iterations,
                    n_linesearch_iterations=solver._n_linesearch_iterations,
                    pcg_threshold=max(solver._pcg_threshold, np.finfo(gs.np_float).eps ** 2),
                    newton_dx_threshold=solver._newton_dx_threshold,
                    linesearch_c=solver._linesearch_c,
                    linesearch_tau=solver._linesearch_tau,
                    # The elimination runs in single precision (see 'FEMCoarseSolve')
                    coarse_pivot_rtol=1e-4,
                )
            unused = {
                IMPLICIT: not is_implicit,
                COROTATED: not config.has_linear_corotated,
                MUSCLE_FIBERS: not (is_implicit and config.has_muscle),
                CORNER_BLOCKS: not config.enable_cooperative_kernels and not is_implicit,
            }
        super().__init__(
            sizes=dict(
                envs=solver._B,
                vertices=solver.n_vertices,
                elements=solver.n_elements,
                vverts=solver.n_vverts,
                surfaces=solver.n_surfaces,
                vfaces=solver.n_vfaces,
                surface_vertices=n_surface_vertices,
                surface_elements=n_surface_elements,
                clusters=len(clusters.clusters_entity_idx),
                clusters_vertices=len(clusters.clusters_vertices_idx),
                clusters_elements=len(clusters.clusters_elements_idx),
                coarse_entities=len(clusters.coarse_entities_cluster_start) - 1,
                coarse_entries=clusters.coarse_entities_entry_start[-1],
            ),
            values=dict(fem_info=info_values),
            env_first={
                (FEMData.fem_state.vertices, FEMData.fem_state.elements, FEMData.fem_state.vverts_render): False,
                (
                    FEMData.fem_state.elements_material,
                    FEMData.fem_state.elements_energy,
                    FEMData.fem_state.vertices_energy,
                    FEMData.fem_state.vertices_solve,
                    FEMData.fem_state.coarse_solve,
                ): True,
            },
            unused=unused,
            requires_grad=solver._requires_grad,
        )


class FEMSolver(GravityMixin, TimeBasedMixin, Solver):
    # Its whole state lives in the arrays 'data' yields (see 'CheckpointMixin')
    is_checkpointable = True

    material_cls = FEM.Base
    is_differentiable = True

    # ------------------------------------------------------------------------------------
    # --------------------------------- Initialization -----------------------------------
    # ------------------------------------------------------------------------------------

    def __init__(self, scene: "Scene", sim: "Simulator", options: FEMOptions) -> None:
        super().__init__(scene, sim, options)

        self._damping = options.damping
        self._use_implicit_solver = options.use_implicit_solver
        # Whether the coupler solves the motion of the implicit solver from the momentum residual at the start of each
        # substep, which spares the free solve (see 'Simulator.build').
        self._is_free_solve_skipped = False
        self._n_newton_iterations = options.n_newton_iterations
        self._newton_dx_threshold = options.newton_dx_threshold
        self._n_pcg_iterations = options.n_pcg_iterations
        # A differentiable scene solves to round-off, which the adjoint of the root of the solve needs to match the finite
        # differences of the simulation
        self._pcg_threshold = options.pcg_threshold
        if self._pcg_threshold is None:
            self._pcg_threshold = 0.0 if self._requires_grad else PCG_THRESHOLD_DEFAULT[gs.np_float]
        self._n_linesearch_iterations = options.n_linesearch_iterations
        self._linesearch_c = options.linesearch_c
        self._linesearch_tau = options.linesearch_tau
        self._damping_alpha = options.damping_alpha
        self._damping_beta = options.damping_beta
        self._max_vertex_constraints = options.max_vertex_constraints

        # use scaled volume for better numerical stability, similar to p_vol_scale in mpm
        self._vol_scale = 1e4

        self.fem_state: FEMState | None = None
        self.fem_info: FEMInfo | None = None
        self.fem_config: FEMConfig | None = None
        self.vertex_constraints: FEMVertexConstraints | None = None
        self.data_manager: FEMData | None = None

    def build(self):
        super().build()

        self._B = self.sim._B

        # The implicit solve adds up every term in a fixed order, so it is reproducible bitwise. On CPU, one thread runs
        # each environment, the environments filling the cores. On GPU, the lanes of a subgroup tile gather the terms
        # of a vertex and a block of threads sums those of an environment.
        enable_cooperative_kernels = gs.backend != gs.cpu

        # The entities holding elements, whose models and muscles the kernels dispatch on, a cloth entity holding
        # surface triangles alone.
        entities_volume = [entity for entity in self._entities if not isinstance(entity.material, FEM.Cloth)]
        materials = [entity.material for entity in entities_volume]
        models = {material.model_id for material in materials}
        # The clusters of each entity of elements (see 'FEMCoarseSolve'). Every entity holds one cluster, then the
        # entities double their clusters in turn as long as theirs keep MIN_CLUSTER_VERTICES vertices, up to
        # MAX_ENTITY_CLUSTERS, and the scene MAX_SCENE_CLUSTERS.
        entities_n_clusters = np.ones(len(entities_volume), dtype=gs.np_int)
        entities_max_clusters = [
            min(MAX_ENTITY_CLUSTERS, entity.n_vertices // MIN_CLUSTER_VERTICES) for entity in entities_volume
        ]
        is_doubled = True
        while is_doubled:
            is_doubled = False
            for i_ent in range(len(entities_volume)):
                n_entity_clusters = entities_n_clusters[i_ent]
                if (
                    2 * n_entity_clusters <= entities_max_clusters[i_ent]
                    and entities_n_clusters.sum() + n_entity_clusters <= MAX_SCENE_CLUSTERS
                ):
                    entities_n_clusters[i_ent] = 2 * n_entity_clusters
                    is_doubled = True
        # One block of threads runs the whole PCG solve of an environment once the environments number half the compute
        # units of the device, their blocks then keeping the device busy: each environment keeps its data close to its
        # block and steps without kernel boundaries. With fewer environments, the device-side loop spreads each
        # environment over the whole device.
        is_pcg_solved_per_env_block = (
            self._use_implicit_solver
            and enable_cooperative_kernels
            and 2 * self._B * get_gpu_cores_per_unit() >= get_gpu_core_count()
        )
        self.fem_config = FEMConfig(
            use_implicit_solver=self._use_implicit_solver,
            enable_cooperative_kernels=enable_cooperative_kernels,
            # The generic kernels outside performance mode serve every scene with the largest bound, sparing a
            # compilation per bound at a small cost for the scenes of small entities.
            max_entity_clusters=MAX_ENTITY_CLUSTERS if gs.use_ndarray else int(entities_n_clusters.max(initial=1)),
            is_pcg_solved_per_env_block=is_pcg_solved_per_env_block,
            has_linesearch=self._n_linesearch_iterations > 0,
            has_damping_alpha=self._damping_alpha > 0.0,
            has_damping_beta=self._damping_beta > gs.EPS,
            has_linear=FEMModel.LINEAR in models,
            has_stable_neohookean=FEMModel.STABLE_NEOHOOKEAN in models,
            has_linear_corotated=FEMModel.LINEAR_COROTATED in models,
            # The elements sharing a model skip the test of their own, except in the solve of an environment per block:
            # the test changes the registers the compiler gives that kernel, and the number of blocks sharing a
            # multiprocessor with them, which runs faster with the test there.
            shared_model=next(iter(models)) if len(models) == 1 and not is_pcg_solved_per_env_block else -1,
            has_muscle=any([isinstance(material, FEM.Muscle) for material in materials]),
        )

        is_active = self.is_active
        # The topology of the elements (see FEMInfo), from the entities
        surface_triangles = np.concatenate(
            [
                np.zeros((0, 3), dtype=gs.np_int),
                *[entity.surface_triangles + entity.v_start for entity in self._entities],
            ]
        )
        el2v = np.concatenate(
            [np.zeros((0, 4), dtype=gs.np_int), *[entity.elems + entity.v_start for entity in entities_volume]]
        )
        elements_idx = np.concatenate(
            [
                np.zeros((0,), dtype=gs.np_int),
                *[np.arange(entity.el_start, entity.el_start + entity.n_elements) for entity in entities_volume],
            ]
        )
        is_vertex_on_surface = np.zeros(self.n_vertices, dtype=bool)
        is_vertex_on_surface[surface_triangles] = True
        (surface_vertices,) = is_vertex_on_surface.nonzero()
        surface_elements = elements_idx[is_vertex_on_surface[el2v].any(axis=1)]

        # The clusters of the vertices of each entity of elements, and the elements touching them (see FEMInfo). The
        # vertices of an entity split in halves at the median of their rest positions along their longest extent, then
        # each half likewise: the rigid motions of compact clusters of balanced sizes span the smooth deformations of
        # the entity.
        vertices_cluster_idx = np.full(self.n_vertices, -1, dtype=gs.np_int)
        n_clusters = 0
        for entity, n_entity_clusters in zip(entities_volume, entities_n_clusters):
            verts_pos = tensor_to_array(entity.init_positions)
            clusters_verts_idx = [np.arange(entity.n_vertices)]
            while len(clusters_verts_idx) < n_entity_clusters:
                clusters_verts_halves = []
                for verts_idx in clusters_verts_idx:
                    cluster_pos = verts_pos[verts_idx]
                    axis = np.argmax(cluster_pos.max(axis=0) - cluster_pos.min(axis=0))
                    verts_idx_sorted = verts_idx[np.argsort(cluster_pos[:, axis], kind="stable")]
                    clusters_verts_halves += np.array_split(verts_idx_sorted, 2)
                clusters_verts_idx = clusters_verts_halves
            for verts_idx in clusters_verts_idx:
                vertices_cluster_idx[entity.v_start + verts_idx] = n_clusters
                n_clusters += 1
        (clustered_vertices,) = (vertices_cluster_idx >= 0).nonzero()
        clusters_vertices_idx = clustered_vertices[np.argsort(vertices_cluster_idx[clustered_vertices], kind="stable")]
        # The pairs of a cluster and an element touching it, sorted by cluster then element
        corners_cluster_idx = vertices_cluster_idx[el2v]
        corners_element_idx = np.broadcast_to(elements_idx[:, None], corners_cluster_idx.shape)
        clusters_elements = np.unique(
            np.stack((corners_cluster_idx, corners_element_idx), axis=-1).reshape((-1, 2)), axis=0
        )
        coarse_entities_entry_start = np.concatenate(([0], np.cumsum((6 * entities_n_clusters) ** 2)))
        clusters = FEMClusters(
            vertices_cluster_idx=vertices_cluster_idx,
            clusters_entity_idx=np.repeat(np.arange(len(entities_volume)), entities_n_clusters),
            clusters_vertices_start=np.searchsorted(
                vertices_cluster_idx[clusters_vertices_idx], np.arange(n_clusters + 1)
            ),
            clusters_vertices_idx=clusters_vertices_idx,
            clusters_elements_start=np.searchsorted(clusters_elements[:, 0], np.arange(n_clusters + 1)),
            clusters_elements_idx=clusters_elements[:, 1],
            coarse_entities_cluster_start=np.concatenate(([0], np.cumsum(entities_n_clusters))),
            coarse_entities_entry_start=coarse_entities_entry_start,
        )
        self.data_manager = FEMData(self, len(surface_vertices), len(surface_elements), clusters)
        self.fem_state = self.data_manager.fem_state
        self.fem_info = self.data_manager.fem_info
        self.register_errors(
            self.data_manager.errno,
            {
                FEMErrorCode.INVALID_VERTEX_NAN: lambda: (
                    "NaN detected in FEM states. Try reducing the time step size or adjusting simulation parameters."
                ),
            },
        )
        self.vertex_constraints = FEMVertexConstraints(unused=True)

        if is_active:
            for entity in self._entities:
                entity._add_to_solver()
            if qd_to_numpy(self.fem_info.has_inverted_elements):
                gs.raise_exception(
                    "The vertices of some tetrahedral elements are ordered incorrectly. Check the input mesh."
                )

            n_vertices = self.n_vertices
            pos = qd_to_numpy(self.fem_state.vertices.pos, row_mask=surface_vertices, col_mask=0, keepdim=False)
            surface_vertices_mapping = np.full(n_vertices, -1, dtype=gs.np_int)
            surface_vertices_mapping[surface_vertices] = np.arange(len(surface_vertices))
            surface_tris_sv_idx = surface_vertices_mapping[surface_triangles]
            # The incident triangles of each surface vertex in increasing triangle index (stable sort of the corners)
            corners_order = np.argsort(surface_tris_sv_idx, axis=None, kind="stable")
            corners_tri_idx = corners_order // 3
            tri2el = qd_to_numpy(self.fem_info.surfaces.tri2el)
            topology_values = [
                (self.fem_info.surface_vertices, surface_vertices),
                (self.fem_info.surface_vert_mass, igl.massmatrix(pos, surface_tris_sv_idx).diagonal()),
                (
                    self.fem_info.surface_vertices_tri_start,
                    np.searchsorted(surface_tris_sv_idx.take(corners_order), np.arange(len(surface_vertices) + 1)),
                ),
                (self.fem_info.surface_vertices_tri_idx, corners_tri_idx),
                (
                    self.fem_info.surface_vertices_tri_mass_scaled,
                    qd_to_numpy(self.fem_info.elements.mass_scaled)[tri2el[corners_tri_idx]],
                ),
                (self.fem_info.surface_elements, surface_elements),
            ]
            if self.fem_config.enable_cooperative_kernels or self._use_implicit_solver:
                # The corners of every element, those of Cloth ones included since their vertices are counted too. The
                # SAP coupler gathers the terms of the vertices of the implicit solver in their order on every backend.
                elements_el2v = qd_to_numpy(self.fem_info.elements.el2v)
                topology_values += [
                    (
                        self.fem_info.vertices_corners_start,
                        np.searchsorted(np.sort(elements_el2v, axis=None), np.arange(n_vertices + 1)),
                    ),
                    (self.fem_info.vertices_corners_idx, np.argsort(elements_el2v, axis=None, kind="stable")),
                ]
            if self._use_implicit_solver:
                topology_values += [
                    (self.fem_info.vertices_cluster_idx, clusters.vertices_cluster_idx),
                    (self.fem_info.clusters_entity_idx, clusters.clusters_entity_idx),
                    (self.fem_info.clusters_vertices_start, clusters.clusters_vertices_start),
                    (self.fem_info.clusters_vertices_idx, clusters.clusters_vertices_idx),
                    (self.fem_info.clusters_elements_start, clusters.clusters_elements_start),
                    (self.fem_info.clusters_elements_idx, clusters.clusters_elements_idx),
                    (self.fem_info.coarse_entities_cluster_start, clusters.coarse_entities_cluster_start),
                    (self.fem_info.coarse_entities_entry_start, clusters.coarse_entities_entry_start),
                ]
            # An empty array keeps the single slot every array is sized to at least, which no kernel reads
            for array, values in topology_values:
                if len(values) == 0:
                    continue
                array.from_numpy(values.astype(gs.np_float if array.dtype == gs.qd_float else gs.np_int, copy=False))

            # Laid out environment-first like the arrays of the solve, whose blocks read the vertices of an environment
            self.vertex_constraints = FEMVertexConstraints(
                sizes=dict(vertices=self.n_vertices, constraint_slots=self._max_vertex_constraints, envs=self._B),
                env_first=True,
            )
            kernel_clear_vertex_constraints(self.vertex_constraints)

        self._build_gravity(self.fem_info.gravity)

    def reset_grad(self) -> None:
        if self._requires_grad:
            qd_zero_grad(self.fem_state.vertices)
            qd_zero_grad(self.fem_state.elements)

    @property
    def is_active(self):
        # Counted from the entities rather than from what build recorded, so the answer stands before it: the rate
        # every solver integrates at is settled from the active ones, and that is settled before any of them builds.
        return self.n_elements > 0

    def add_entity(
        self, idx, material, morph, surface, visualize_contact=False, name: str | None = None, desc=None
    ) -> "FEMEntity":
        entity = FEMEntity(
            scene=self._scene,
            solver=self,
            material=material,
            morph=morph,
            surface=surface,
            idx=idx,
            v_start=self.n_vertices,
            el_start=self.n_elements,
            s_start=self.n_surfaces,
            vvert_start=self.n_vverts,
            vface_start=self.n_vfaces,
            name=name,
        )
        self._entities.append(entity)
        return entity

    def add_elements(
        self,
        material: FEM.Elastic,
        v_start: int,
        el_start: int,
        s_start: int,
        verts: np.ndarray,
        elems: np.ndarray,
        tri2v: np.ndarray,
        tri2el: np.ndarray,
    ) -> None:
        """Add the vertices, elements and surface triangles of an entity at rest, with its material parameters."""
        is_muscle = isinstance(material, FEM.Muscle)
        kernel_add_elements(
            v_start,
            el_start,
            s_start,
            model=int(material.model_id),
            mu=material.mu,
            lam=material.lam,
            rho=material.rho,
            friction_mu=material.friction_mu,
            muscle_stiffness=material.stiffness if is_muscle else 0.0,
            verts=verts,
            elems=elems,
            tri2v=tri2v,
            tri2el=tri2el,
            fem_state=self.fem_state,
            fem_info=self.fem_info,
            is_muscle=is_muscle,
        )

    def add_cloth(self, v_start: int, s_start: int, verts: np.ndarray, tri2v: np.ndarray, tri2el: np.ndarray) -> None:
        """
        Add cloth vertices and surface triangles to the solver, for position tracking and coupling only.

        Cloth elements and mass are owned by the incremental potential contact (IPC) coupler, so the vertex info holds
        placeholder values and each surface triangle references itself as element.
        """
        kernel_add_cloth(v_start, s_start, verts, tri2v, tri2el, self.fem_state, self.fem_info)

    def add_vverts(
        self,
        vvert_start: int,
        vface_start: int,
        v_start: int,
        verts_idx: np.ndarray,
        uvs: np.ndarray,
        vfaces: np.ndarray,
    ) -> None:
        """Add the visual vertices and faces of a visual geom, the former standing for the given simulated vertices."""
        kernel_add_vverts(vvert_start, vface_start, v_start, verts_idx, uvs, vfaces, self.fem_info)

    # ------------------------------------------------------------------------------------
    # ------------------------------------ stepping --------------------------------------
    # ------------------------------------------------------------------------------------

    def substep_pre_coupling(self) -> None:
        if not self.is_active:
            return
        # The IPC coupler simulates the FEM entities itself. Imported here to avoid a circular dependency, the couplers
        # depending on the solvers.
        from genesis.engine.couplers import IPCCoupler

        if isinstance(self._sim.coupler, IPCCoupler):
            return
        links_state = self._sim.rigid_solver.dyn_state.links
        if self._use_implicit_solver:
            kernel_precompute_material_data(self.fem_state, self.fem_info, self.fem_config)
            kernel_init_pos_and_inertia(self.fem_state, links_state, self.fem_info, self.vertex_constraints)
            # The SAP coupler solves the motion from the momentum residual instead (see 'Simulator.build')
            if not self._is_free_solve_skipped:
                # The solve is reversed through the adjoint of its root, which spares reversing its iterations
                with custom_backward(self._implicit_solve_grad):
                    if self._requires_grad and gs.use_ndarray:
                        self._solve_on_host(is_adjoint=False)
                    else:
                        loops = self.fem_state.solve_loops
                        kernel_batch_solve(
                            loops.newton_counter,
                            loops.pcg_counter,
                            loops.linesearch_counter,
                            self.fem_state,
                            self.fem_info,
                            self.vertex_constraints,
                            self.fem_config,
                        )
            kernel_setup_pos_vel(self.fem_state, self.fem_info)
        else:
            kernel_init_vel(self.fem_state)
            kernel_compute_vel(self.fem_state, self.fem_info, self.fem_config)
            kernel_apply_uniform_force(self.fem_state, self.fem_info)
            kernel_apply_soft_constraints(self.fem_state, links_state, self.fem_info, self.vertex_constraints)

    def substep_post_coupling(self) -> None:
        if not self.is_active:
            return
        kernel_compute_pos(self.fem_state, self.fem_info, self._errno)
        if not self._use_implicit_solver:
            kernel_apply_hard_constraints(
                self.fem_state, self._sim.rigid_solver.dyn_state.links, self.vertex_constraints, self._errno
            )
        # A kernel rather than a zero-copy view, since the launch recorder replays kernels only
        kernel_copy_next_to_curr(self.fem_state)

    def _implicit_solve_grad(self) -> None:
        """
        Back-propagate the adjoint of the end positions of a substep of the implicit solver to its inputs.

        The end positions x+ are the root of the forces R(x+; x, v, actu), whose Jacobian with respect to x+ is minus
        the Hessian H of the incremental energy, symmetric and constant since every material the implicit solver
        supports has an energy quadratic in x+. The adjoint solve H lambda = g of the adjoint g of x+ runs the
        preconditioned conjugate gradient (PCG) of the solve, then the autodiff of the forces seeded with lambda adds
        lambda^T dR/d(input) to the adjoint of every input but the data the materials compute from the start of the
        substep, which autodiff takes as constant and the closed-form reverse of the materials carries to the start
        positions instead. A constrained vertex has no force, so its target takes its adjoint plus the pull of lambda.
        """
        if gs.use_ndarray:
            self._solve_on_host(is_adjoint=True)
        else:
            args = (self.fem_state, self.fem_info, self.vertex_constraints, self.fem_config)
            kernel_solve_adjoint(self.fem_state.solve_loops.pcg_counter, *args)
        kernel_compute_implicit_force.grad(self.fem_state, self.fem_info, self.vertex_constraints, self.fem_config)
        kernel_material_data_grad(self.fem_state, self.fem_info, self.vertex_constraints, self.fem_config)

    def _solve_on_host(self, is_adjoint: bool) -> None:
        # FIXME: quadrants#946 - a graph refuses ndarrays owning a gradient, so the loops of 'kernel_batch_solve' run
        # on the host for such scenes, reading the flag of each loop after every iteration.
        loops = self.fem_state.solve_loops
        args = (
            loops.newton_counter,
            loops.pcg_counter,
            loops.linesearch_counter,
            self.fem_state,
            self.fem_info,
            self.vertex_constraints,
            self.fem_config,
        )
        # The stages go as native ints, since a template argument other than a native scalar is weakly referenced
        if not is_adjoint:
            kernel_solve_stage(*args, stage=int(FEMSolveStage.INIT_NEWTON))
        # Each loop runs its body once before testing its flag, as a graph do-while does
        n_newton_iterations = 1 if is_adjoint else max(self._n_newton_iterations, 1)
        for i_newton in range(n_newton_iterations):
            stage = FEMSolveStage.INIT_ADJOINT if is_adjoint else FEMSolveStage.NEWTON_HEAD
            kernel_solve_stage(*args, stage=int(stage))
            is_looping = True
            while is_looping:
                kernel_solve_stage(*args, stage=int(FEMSolveStage.PCG_ITER))
                is_looping = not self.fem_config.is_pcg_solved_per_env_block and qd_to_numpy(loops.pcg_counter)
            if is_adjoint:
                kernel_solve_stage(*args, stage=int(FEMSolveStage.SEED_ADJOINT))
            elif self.fem_config.has_linesearch:
                kernel_solve_stage(*args, stage=int(FEMSolveStage.INIT_LINESEARCH))
                is_looping = True
                while is_looping:
                    kernel_solve_stage(*args, stage=int(FEMSolveStage.LINESEARCH_ITER))
                    is_looping = qd_to_numpy(loops.linesearch_counter)
            else:
                kernel_solve_stage(*args, stage=int(FEMSolveStage.SKIP_LINESEARCH))
            if not is_adjoint:
                kernel_solve_stage(*args, stage=int(FEMSolveStage.END_NEWTON))
                if i_newton + 1 < n_newton_iterations and not qd_to_numpy(loops.newton_counter):
                    break

    # ------------------------------------------------------------------------------------
    # ----------------------------------- gradient ---------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def data(self) -> Iterator[DataItem | ConfigItem]:
        yield from iter_data(self.data_manager, "fem_info")
        yield from iter_data(self.data_manager, "fem_state")
        yield from iter_data(self, "vertex_constraints")
        yield from iter_data(self.data_manager, "errno", DataKind.STATE)
        yield from iter_data(self, "fem_config")

    # ------------------------------------------------------------------------------------
    # --------------------------------------- io -----------------------------------------
    # ------------------------------------------------------------------------------------

    def get_state_render(self, f: int | None = None):
        """
        Refresh and return the render geometry of every visual geom, laid out contiguously.

        Parameters
        ----------
        f : int, optional
            Deprecated and ignored.

        Returns
        -------
        tuple
            (vverts_pos, vverts_uvs, vfaces_indices) - environment-offset render vertex positions with shape
            (n_vverts, B), their UV coordinates, and the render triangles in global render vertex space.
        """
        if f is not None:
            warn_once(
                "The argument of 'get_state_render' is deprecated and ignored. Call 'get_state_render()' instead."
            )
        if not self.is_active or self.n_vverts == 0:
            return None, None, None

        kernel_update_vverts_render(self.fem_state, self.fem_info)
        return self.fem_state.vverts_render.pos, self.fem_info.vverts_uvs, self.fem_info.vfaces_indices

    def get_forces(self):
        """
        Get the internal elastic force on every vertex, computed from the current state.

        The force of each element follows the material model of the solver in use, and the forces of a body sum to zero.
        Damping, gravity and contact forces are left out. With the incremental potential contact (IPC) coupler, which
        simulates the FEM entities itself, it raises an exception.

        Returns
        -------
        forces : torch.Tensor
            The force on each vertex of each environment.
        """
        if not self.is_active:
            return None
        # Imported here to avoid a circular dependency (see 'substep_pre_coupling')
        from genesis.engine.couplers import IPCCoupler

        # The IPC coupler simulates the FEM entities with its own models, and leaves the elements of Cloth ones empty.
        if isinstance(self._sim.coupler, IPCCoupler):
            gs.raise_exception("The internal forces of FEM entities are not available with the IPC coupler.")

        # The linear corotated model takes the rotation of each element from the current state
        if self._use_implicit_solver:
            kernel_precompute_material_data(self.fem_state, self.fem_info, self.fem_config)
        forces = torch.zeros((self._B, self.n_vertices, 3), dtype=gs.tc_float, device=gs.device)
        kernel_get_forces(forces, self.fem_state, self.fem_info, self.vertex_constraints, self.fem_config)
        return forces

    @gs.assert_built
    @mutates(StateChange.GEOMETRY)
    def set_verts_pos(self, poss, verts_idx=None, envs_idx=None):
        """
        Set the position of some vertices.

        Parameters
        ----------
        poss : array_like, shape ([M,] [N,] 3)
            Target position of each vertex.
        verts_idx : None | int | array_like, shape ([M,] N), optional
            Global index of the vertices, once for every environment or one row per environment. If None, all vertices
            will be considered. Defaults to None.
        envs_idx : None | int | array_like, shape (M,), optional
            The indices of the environments to set. If None, all environments will be considered. Defaults to None.
        """
        self._write_data(
            self.fem_state.vertices.pos, poss, verts_idx, self.n_vertices, "verts_idx", envs_idx, element_shape=(3,)
        )
        if gs.use_zerocopy and gs.backend == gs.metal:
            torch.mps.synchronize()

    def get_verts_pos(self, verts_idx=None, envs_idx=None):
        """Get the position of some vertices, the same in every environment (see 'set_verts_pos')."""
        tensor = self._read_data(self.fem_state.vertices.pos, envs_idx, verts_idx, copy=True)
        return tensor[0] if self._scene.n_envs == 0 else tensor

    def get_verts_vel(self, verts_idx=None, envs_idx=None):
        """Get the velocity of some vertices, the same in every environment (see 'set_verts_pos')."""
        tensor = self._read_data(self.fem_state.vertices.vel, envs_idx, verts_idx, copy=True)
        return tensor[0] if self._scene.n_envs == 0 else tensor

    @gs.assert_built
    @mutates(StateChange.DYNAMICS)
    def set_verts_vel(self, vels, verts_idx=None, envs_idx=None):
        """Set the velocity of some vertices (see 'set_verts_pos')."""
        self._write_data(
            self.fem_state.vertices.vel, vels, verts_idx, self.n_vertices, "verts_idx", envs_idx, element_shape=(3,)
        )
        if gs.use_zerocopy and gs.backend == gs.metal:
            torch.mps.synchronize()

    @gs.assert_built
    @mutates()
    def set_elements_actu(self, actus, elements_idx=None, envs_idx=None, *, n_groups=1):
        """
        Set the actuation of some elements, each element taking the actuation of its muscle group.

        Parameters
        ----------
        actus : array_like, shape ([M,] [N,] [n_groups,])
            Actuation of each muscle group, for each element.
        elements_idx : None | int | array_like, shape ([M,] N), optional
            Global index of the elements (see 'set_verts_pos'). If None, all elements will be considered.
        envs_idx : None | int | array_like, shape (M,), optional
            The indices of the environments to set. If None, all environments will be considered. Defaults to None.
        n_groups : int, optional
            The number of muscle groups. Defaults to 1.
        """
        actus, elements_idx, envs_idx = self._sanitize_io_variables(
            actus, elements_idx, self.n_elements, "elements_idx", envs_idx, (n_groups,), batched_idx=True
        )
        kernel_set_actu(
            elements_idx, envs_idx, actus, self.fem_info.elements.muscle_group, self.fem_state.elements.actu
        )

    @mutates(StateChange.MODEL)
    def set_elements_muscle_group(self, muscle_group, elements_idx=None):
        """Set the muscle group of some elements, shared by every environment."""
        self._write_data(
            self.fem_info.elements.muscle_group,
            muscle_group,
            elements_idx,
            self.n_elements,
            "elements_idx",
            dtype=gs.tc_int,
            is_batched=False,
        )
        if gs.use_zerocopy and gs.backend == gs.metal:
            torch.mps.synchronize()

    @mutates(StateChange.MODEL)
    def set_elements_muscle_direction(self, muscle_direction, elements_idx=None):
        """Set the unit direction of the muscle fibers of some elements, shared by every environment."""
        self._write_data(
            self.fem_info.elements.muscle_direction,
            muscle_direction,
            elements_idx,
            self.n_elements,
            "elements_idx",
            element_shape=(3,),
            is_batched=False,
        )
        if gs.use_zerocopy and gs.backend == gs.metal:
            torch.mps.synchronize()

    def get_elements_el2v(self, elements_idx=None):
        """Get the global index of the four vertices of some elements."""
        return self._read_data(self.fem_info.elements.el2v, None, elements_idx, copy=True, is_batched=False)

    @gs.assert_built
    @mutates(StateChange.MODEL)
    def set_vertex_constraints(
        self, verts_idx=None, target_poss=None, link_idx=-1, is_soft_constraint=False, stiffness=0.0, envs_idx=None
    ):
        """
        Constrain some vertices to target positions, or to follow a rigid link.

        Parameters
        ----------
        verts_idx : None | int | array_like, shape ([M,] N), optional
            Global index of the vertices (see 'set_verts_pos'). If None, all vertices will be considered.
        target_poss : None | array_like, shape ([M,] [N,] 3), optional
            Target position of each vertex. If None, the current positions are used. Defaults to None.
        link_idx : int, optional
            Global index of the rigid link the vertices follow, keeping their offset to it, or -1 for none.
        is_soft_constraint : bool, optional
            Whether to pull the vertices toward their targets with critically damped springs, rather than setting their
            position and zeroing their velocity. A spring applies the force -k (x - x_t) - 2 sqrt(k m) (v - v_t) to a
            vertex of mass m, x_t and v_t being the position and velocity of its target. The implicit solver holds the
            vertices at their targets instead. Defaults to False.
        stiffness : float, optional
            Stiffness k of the springs of a soft constraint, in N/m. Defaults to 0.0.
        envs_idx : None | int | array_like, shape (M,), optional
            The indices of the environments to set. If None, all environments will be considered. Defaults to None.
        """
        # Imported here to avoid a circular dependency (see 'substep_pre_coupling')
        from genesis.engine.couplers import IPCCoupler

        if isinstance(self._sim.coupler, IPCCoupler):
            gs.raise_exception("Vertex constraints are not supported by the IPC coupler.")

        is_target_current = target_poss is None
        target_poss, verts_idx, envs_idx = self._sanitize_io_variables(
            target_poss, verts_idx, self.n_vertices, "verts_idx", envs_idx, (3,), batched_idx=True
        )
        if is_target_current:
            kernel_get_values(verts_idx, envs_idx, target_poss, self.fem_state.vertices.pos)

        if link_idx < 0:
            link_init_pos = torch.zeros((len(envs_idx), 3), dtype=gs.tc_float, device=gs.device)
            link_init_quat = torch.zeros((len(envs_idx), 4), dtype=gs.tc_float, device=gs.device)
        else:
            rigid_solver = self._sim.rigid_solver
            links_envs_idx = envs_idx if self._scene.n_envs > 0 else None
            # The attachment is a configuration of the constraint, which carries no gradient
            with torch.no_grad():
                link_init_pos = rigid_solver.get_links_pos(links_idx=link_idx, envs_idx=links_envs_idx)[..., 0, :]
                link_init_quat = rigid_solver.get_links_quat(links_idx=link_idx, envs_idx=links_envs_idx)[..., 0, :]
            link_init_pos = broadcast_tensor(link_init_pos, gs.tc_float, (len(envs_idx), 3)).contiguous()
            link_init_quat = broadcast_tensor(link_init_quat, gs.tc_float, (len(envs_idx), 4)).contiguous()

        # Checked before any environment is written, so that the vertices are constrained in every selected environment
        # or in none. A vertex given several times takes a single slot, the earlier occurrences being -1 when targets are
        # given (see 'drop_repeated_index') and repeated otherwise.
        vertices_slot = qd_to_torch(self.vertex_constraints.vertices_slot, transpose=True)
        envs_verts_idx = envs_idx[:, None].expand_as(verts_idx)
        is_new = (verts_idx >= 0) & (vertices_slot[envs_verts_idx, verts_idx.clamp(min=0)] < 0)
        rows_idx = torch.arange(len(envs_idx), device=gs.device)[:, None].expand_as(verts_idx)
        is_new_vert = torch.zeros((len(envs_idx), self.n_vertices), dtype=torch.bool, device=gs.device)
        is_new_vert.index_put_((rows_idx, verts_idx.clamp(min=0)), is_new, accumulate=True)
        envs_n_constraints = qd_to_torch(self.vertex_constraints.envs_n_constraints, transpose=True)[envs_idx]
        if (envs_n_constraints + is_new_vert.sum(dim=-1) > self._max_vertex_constraints).any():
            gs.raise_exception(
                "No room left for these vertex constraints. Please increase the value of "
                f"{type(self).__name__}'s option 'max_vertex_constraints' ({self._max_vertex_constraints})."
            )

        kernel_set_vertex_constraints(
            verts_idx,
            envs_idx,
            link_idx,
            stiffness,
            target_poss,
            link_init_pos,
            link_init_quat,
            self.fem_state,
            self.vertex_constraints,
            is_soft_constraint,
        )

    @mutates(StateChange.MODEL)
    def update_constraint_targets(self, target_poss, verts_idx=None, envs_idx=None):
        """Update the target position of some constrained vertices (see 'set_vertex_constraints')."""
        if target_poss is None:
            gs.raise_exception("The target positions of the constraints must be specified.")
        target_poss, verts_idx, envs_idx = self._sanitize_io_variables(
            target_poss, verts_idx, self.n_vertices, "verts_idx", envs_idx, (3,), batched_idx=True
        )
        kernel_update_constraint_targets(verts_idx, envs_idx, target_poss, self.vertex_constraints)

    @mutates(StateChange.MODEL)
    def remove_vertex_constraints(self, verts_idx=None, envs_idx=None):
        """Remove the constraints of some vertices (see 'set_vertex_constraints'), of all of them if None."""
        _, verts_idx, envs_idx = self._sanitize_io_variables(
            None, verts_idx, self.n_vertices, "verts_idx", envs_idx, batched_idx=True, skip_allocation=True
        )
        kernel_remove_vertex_constraints(verts_idx, envs_idx, self.vertex_constraints)

    # ------------------------------------------------------------------------------------
    # ----------------------------------- properties -------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def damping(self) -> float:
        return self._damping

    @property
    def n_vertices(self) -> int:
        return sum([entity.n_vertices for entity in self._entities])

    @property
    def n_elements(self) -> int:
        return sum([entity.n_elements for entity in self._entities])

    @property
    def n_surfaces(self) -> int:
        return sum([entity.n_surfaces for entity in self.entities])

    @property
    def n_vverts(self) -> int:
        return sum([entity.n_vverts for entity in self._entities])

    @property
    def n_vfaces(self) -> int:
        return sum([entity.n_vfaces for entity in self._entities])

    @property
    def vol_scale(self) -> float:
        return self._vol_scale

    @property
    def n_surface_vertices(self) -> int:
        return self.fem_info.surface_vertices.shape[0]

    @property
    def n_surface_elements(self) -> int:
        return self.fem_info.surface_elements.shape[0]


# ------------------------------------------------------------------------------------
# ------------------------------------ materials -------------------------------------
# ------------------------------------------------------------------------------------

# The materials of the elements are dispatched at compile time on the constitutive models present in the scene (see
# FEMConfig), each element taking the branch of its own model at runtime. Compiling a model into a kernel outside the
# models it supports raises as the kernel compiles.


@qd.func
def func_element_model(i_e: int, fem_info: FEMInfo, fem_config: qd.template()):
    """Return the constitutive model of an element, a compile-time constant when all the elements share it."""
    model = gs.qd_int(fem_config.shared_model)
    if qd.static(fem_config.shared_model < 0):
        model = fem_info.elements.model[i_e]
    return model


@qd.func
def func_element_stress(
    i_e: int, J: float, F: qd.types.matrix(3, 3), actu: float, fem_info: FEMInfo, fem_config: qd.template()
):
    """Return the first Piola-Kirchhoff stress of an element under the explicit solver."""
    mu = fem_info.elements.mu[i_e]
    lam = fem_info.elements.lam[i_e]
    model = func_element_model(i_e, fem_info, fem_config)
    stress = qd.Matrix.zero(gs.qd_float, 3, 3)
    if qd.static(fem_config.has_linear):
        if model == FEMModel.LINEAR:
            stress = func_linear_stress(mu, lam, F)
    if qd.static(fem_config.has_stable_neohookean):
        if model == FEMModel.STABLE_NEOHOOKEAN:
            stress = func_stable_neohookean_stress(mu, lam, J, F)
    if qd.static(fem_config.has_linear_corotated):
        gs.raise_exception("Linear corotated stress update is not implemented yet.")
    if qd.static(fem_config.has_muscle):
        if fem_info.elements.is_muscle[i_e]:
            stress += func_muscle_stress(
                fem_info.elements.muscle_stiffness[i_e],
                actu,
                F,
                fem_info.elements.muscle_direction[i_e],
                fem_info.fiber_length_eps[None],
            )
    return stress


@qd.func
def func_element_energy_gradient(
    i_e: int,
    i_b: int,
    F: qd.types.matrix(3, 3),
    actu: float,
    fem_state: FEMState,
    fem_info: FEMInfo,
    fem_config: qd.template(),
):
    """Return the energy density of an element and its gradient with respect to the deformation gradient."""
    mu = fem_info.elements.mu[i_e]
    lam = fem_info.elements.lam[i_e]
    model = func_element_model(i_e, fem_info, fem_config)
    energy = gs.qd_float(0.0)
    gradient = qd.Matrix.zero(gs.qd_float, 3, 3)
    if qd.static(fem_config.has_linear):
        if model == FEMModel.LINEAR:
            energy, gradient = func_linear_energy_gradient(mu, lam, F)
    if qd.static(fem_config.has_stable_neohookean):
        gs.raise_exception("Gradient computation is not implemented for stable_neohookean model.")
    if qd.static(fem_config.has_linear_corotated):
        if model == FEMModel.LINEAR_COROTATED:
            energy, gradient = func_linear_corotated_energy_gradient(
                mu, lam, fem_state.elements_material.rotation[i_e, i_b], F
            )
    if qd.static(fem_config.has_muscle):
        if fem_info.elements.is_muscle[i_e]:
            stress = func_muscle_fiber_stress(
                fem_info.elements.muscle_stiffness[i_e],
                actu,
                fem_info.elements.muscle_direction[i_e],
                fem_state.elements_material.fiber_dir[i_e, i_b],
            )
            energy = energy + (stress * F).sum()
            gradient = gradient + stress
    return energy, gradient


@qd.func
def func_element_energy(
    i_e: int,
    i_b: int,
    J: float,
    F: qd.types.matrix(3, 3),
    actu: float,
    fem_state: FEMState,
    fem_info: FEMInfo,
    fem_config: qd.template(),
):
    """Return the energy density of an element."""
    mu = fem_info.elements.mu[i_e]
    lam = fem_info.elements.lam[i_e]
    model = func_element_model(i_e, fem_info, fem_config)
    energy = gs.qd_float(0.0)
    if qd.static(fem_config.has_linear):
        if model == FEMModel.LINEAR:
            energy = func_linear_energy(mu, lam, F)
    if qd.static(fem_config.has_stable_neohookean):
        if model == FEMModel.STABLE_NEOHOOKEAN:
            energy = func_stable_neohookean_energy(mu, lam, J, F)
    if qd.static(fem_config.has_linear_corotated):
        if model == FEMModel.LINEAR_COROTATED:
            energy = func_linear_corotated_energy(mu, lam, fem_state.elements_material.rotation[i_e, i_b], F)
    if qd.static(fem_config.has_muscle):
        if fem_info.elements.is_muscle[i_e]:
            fiber_dir = fem_state.elements_material.fiber_dir[i_e, i_b]
            energy = energy + fem_info.elements.muscle_stiffness[i_e] * actu * fiber_dir.dot(
                F @ fem_info.elements.muscle_direction[i_e]
            )
    return energy


@qd.func
def func_element_hessian_product(
    i_e: int,
    i_b: int,
    dF: qd.types.matrix(3, 3),
    fem_state: FEMState,
    fem_info: FEMInfo,
    fem_config: qd.template(),
):
    """
    Return the product of the Hessian of the energy density of an element with a change of its deformation.

    The implicit solver takes this product in place of the Hessian itself.
    """
    mu = fem_info.elements.mu[i_e]
    lam = fem_info.elements.lam[i_e]
    model = func_element_model(i_e, fem_info, fem_config)
    dP = qd.Matrix.zero(gs.qd_float, 3, 3)
    if qd.static(fem_config.has_linear):
        if model == FEMModel.LINEAR:
            dP = func_linear_hessian_product(mu, lam, dF)
    if qd.static(fem_config.has_stable_neohookean):
        gs.raise_exception("Hessian computation is not implemented for stable_neohookean model.")
    if qd.static(fem_config.has_linear_corotated):
        if model == FEMModel.LINEAR_COROTATED:
            # The Hessian depends on the rotation of the element alone, taken at the start of the substep
            dP = func_linear_corotated_hessian_product(mu, lam, fem_state.elements_material.rotation[i_e, i_b], dF)
    return dP


@qd.func
def func_element_precompute(
    i_e: int,
    i_b: int,
    F: qd.types.matrix(3, 3),
    fem_state: FEMState,
    fem_info: FEMInfo,
    fem_config: qd.template(),
):
    """Compute the data the material of an element takes from its deformation at the start of a substep."""
    if qd.static(fem_config.has_linear_corotated):
        if func_element_model(i_e, fem_info, fem_config) == FEMModel.LINEAR_COROTATED:
            fem_state.elements_material.rotation[i_e, i_b] = func_linear_corotated_rotation(
                F, fem_info.rotation_atol[None]
            )
    if qd.static(fem_config.has_muscle):
        if fem_info.elements.is_muscle[i_e]:
            fem_state.elements_material.fiber_dir[i_e, i_b] = func_muscle_fiber_dir(
                F, fem_info.elements.muscle_direction[i_e], fem_info.fiber_length_eps[None]
            )


@qd.func
def func_element_precompute_grad(
    i_e: int,
    i_b: int,
    F_start: qd.types.matrix(3, 3),
    F: qd.types.matrix(3, 3),
    dF_damping: qd.types.matrix(3, 3),
    gradient_grad: qd.types.matrix(3, 3),
    actu: float,
    fem_state: FEMState,
    fem_info: FEMInfo,
    fem_config: qd.template(),
):
    """
    Return the adjoint of the deformation gradient at the start of a substep that the data of the material carries.

    It is the reverse of 'func_element_precompute', from the adjoint `gradient_grad` of the energy gradient at the end
    of the substep, which the data enters along with the change `dF_damping` of the deformation gradient driving the
    stiffness damping.
    """
    F_start_grad = qd.Matrix.zero(gs.qd_float, 3, 3)
    if qd.static(fem_config.has_linear_corotated):
        if func_element_model(i_e, fem_info, fem_config) == FEMModel.LINEAR_COROTATED:
            F_start_grad = func_linear_corotated_rotation_grad(
                fem_info.elements.mu[i_e],
                fem_info.elements.lam[i_e],
                fem_state.elements_material.rotation[i_e, i_b],
                F_start,
                F,
                dF_damping,
                gradient_grad,
            )
    if qd.static(fem_config.has_muscle):
        if fem_info.elements.is_muscle[i_e]:
            F_start_grad = F_start_grad + func_muscle_fiber_dir_grad(
                fem_info.elements.muscle_stiffness[i_e],
                actu,
                F_start,
                fem_info.elements.muscle_direction[i_e],
                gradient_grad,
                fem_info.fiber_length_eps[None],
            )
    return F_start_grad


# ------------------------------------------------------------------------------------
# ------------------------------------- elements -------------------------------------
# ------------------------------------------------------------------------------------


@qd.func
def func_element_J_F(i_e: int, i_b: int, verts_pos: qd.Tensor, fem_info: FEMInfo):
    """Return the determinant J and the deformation gradient F of an element, its vertices at `verts_pos`."""
    i_vs = fem_info.elements.el2v[i_e]
    pos_v3 = verts_pos[i_vs[3], i_b]
    D = qd.Matrix.cols([verts_pos[i_vs[i_k], i_b] - pos_v3 for i_k in qd.static(range(3))])
    F = D @ fem_info.elements.B[i_e]
    J = F.determinant()
    return J, F


@qd.func
def func_element_internal_force(i_e: int, i_b: int, fem_state: FEMState, fem_info: FEMInfo, fem_config: qd.template()):
    """
    Return the elastic force of an element on each of its vertices, one per column, from the stress its solver uses.

    The columns sum to zero, so the forces conserve momentum once each vertex divides its own by its own mass.
    """
    J, F = func_element_J_F(i_e, i_b, fem_state.vertices.pos, fem_info)
    actu = fem_state.elements.actu[i_e, i_b]
    stress = qd.Matrix.zero(gs.qd_float, 3, 3)
    if qd.static(fem_config.use_implicit_solver):
        _, stress = func_element_energy_gradient(i_e, i_b, F, actu, fem_state, fem_info, fem_config)
    else:
        stress = func_element_stress(i_e, J, F, actu, fem_info, fem_config)
    H = -fem_info.elements.V[i_e] * stress @ fem_info.elements.B[i_e].transpose()
    return qd.Matrix.cols([H[:, 0], H[:, 1], H[:, 2], -H[:, 0] - H[:, 1] - H[:, 2]])


@qd.func
def func_element_displacement(i_e: int, i_b: int, fem_state: FEMState, fem_info: FEMInfo):
    """Return the displacement of each vertex of an element over a substep, one per column."""
    i_vs = fem_info.elements.el2v[i_e]
    return qd.Matrix.cols(
        [
            fem_state.vertices.pos_next[i_vs[i_k], i_b] - fem_state.vertices.pos[i_vs[i_k], i_b]
            for i_k in qd.static(range(4))
        ]
    )


@qd.func
def func_element_mapping_matrix(i_e: int, fem_info: FEMInfo):
    """
    Return the matrix S mapping the displacement of the vertices of an element to the change of its deformation
    gradient.

    The constrained vertices take part in the deformation like the others: their motion strains the element, and the
    forces the element puts on them are dropped where they are gathered (see 'func_is_vertex_free').
    """
    B = fem_info.elements.B[i_e]
    S = qd.Matrix.zero(gs.qd_float, 4, 3)
    S[:3, :] = B
    S[3, :] = -B[0, :] - B[1, :] - B[2, :]
    return S


@qd.func
def func_elastic_product(
    i_e: int,
    i_b: int,
    S: qd.types.matrix(4, 3),
    x: qd.types.matrix(3, 4),
    fem_state: FEMState,
    fem_info: FEMInfo,
    fem_config: qd.template(),
):
    """
    Return the change of the deformation gradient of an element along a displacement, and its energy Hessian product.

    `x` holds the displacement of each vertex of the element in its columns, and `S` maps it to the deformation gradient
    (see 'func_element_mapping_matrix'). The second returned matrix is the product of the Hessian of the energy density
    of the element with the first one.
    """
    dF = x @ S
    dP = func_element_hessian_product(i_e, i_b, dF, fem_state, fem_info, fem_config)
    return dF, dP


@qd.func
def func_element_energy_gradient_damped(
    i_e: int, i_b: int, fem_state: FEMState, fem_info: FEMInfo, fem_config: qd.template()
):
    """
    Return the energy density of an element at the end of a substep of the implicit solver, and its damped gradient.

    The gradient is taken with respect to the deformation gradient, plus the stress of the Rayleigh stiffness damping
    for the displacement over the substep.
    """
    # The placeholders of the values left unused are named apart, since autodiff gives a reused name one type alone
    J, F = func_element_J_F(i_e, i_b, fem_state.vertices.pos_next, fem_info)
    actu = fem_state.elements.actu[i_e, i_b]
    energy, gradient = func_element_energy_gradient(i_e, i_b, F, actu, fem_state, fem_info, fem_config)
    if qd.static(fem_config.has_damping_beta):
        S = func_element_mapping_matrix(i_e, fem_info)
        x = func_element_displacement(i_e, i_b, fem_state, fem_info)
        dF, dP = func_elastic_product(i_e, i_b, S, x, fem_state, fem_info, fem_config)
        gradient = gradient + fem_info.damping_beta_over_dt[None] * dP
    return energy, gradient


@qd.func
def func_vertex_inertia_force(
    i_v: int,
    i_b: int,
    has_constraints: bool,
    fem_state: FEMState,
    fem_info: FEMInfo,
    vertex_constraints: FEMVertexConstraints,
):
    """
    Return the force of the inertia and of the Rayleigh mass damping on a vertex at the end of a substep of the
    implicit solver.

    A constrained vertex sits at its target, which the damping of its jump would otherwise pull it from.
    """
    pos = fem_state.vertices.pos_next[i_v, i_b]
    force = -fem_info.vertices.mass_over_dt2[i_v] * (
        (pos - fem_state.vertices_energy.inertia[i_v, i_b])
        + (pos - fem_state.vertices.pos[i_v, i_b]) * fem_info.damping_alpha_dt[None]
    )
    if not func_is_vertex_free(i_v, i_b, has_constraints, vertex_constraints):
        force = qd.Vector.zero(gs.qd_float, 3)
    return force


@qd.func
def func_constraint_target(i_s: int, i_b: int, links_state: LinksState, vertex_constraints: FEMVertexConstraints):
    """
    Return the target of the vertex constraint in a slot of an environment.

    The target of a vertex following a link is computed from the pose of the link wherever it is used, so that the
    gradient flows from the link to the vertex.
    """
    target_pos = vertex_constraints.target_pos[i_s, i_b]
    i_l = vertex_constraints.link_idx[i_s, i_b]
    if i_l >= 0:
        offset_quat = qd_transform_quat_by_quat(
            qd_inv_quat(vertex_constraints.link_init_quat[i_s, i_b]), links_state.quat[i_l, i_b]
        )
        offset_pos = qd_transform_by_quat(vertex_constraints.link_offset_pos[i_s, i_b], offset_quat)
        target_pos = links_state.pos[i_l, i_b] + offset_pos
    return target_pos


# ------------------------------------------------------------------------------------
# --------------------------------- explicit solver ----------------------------------
# ------------------------------------------------------------------------------------


@qd.kernel(fastcache=True)
def kernel_init_vel(fem_state: FEMState):
    """Start the velocities of the vertices at the end of a substep of the explicit solver from those at its start."""
    for i_v, i_b in qd.ndrange(*fem_state.vertices.vel.shape, axes=qd.static(eoa(fem_state.vertices.vel))):
        fem_state.vertices.vel_next[i_v, i_b] = fem_state.vertices.vel[i_v, i_b]


@qd.kernel(fastcache=True)
def kernel_compute_vel(fem_state: FEMState, fem_info: FEMInfo, fem_config: qd.template()):
    """Add the velocity the elastic forces of the elements give their vertices over a substep."""
    for i_e, i_b in qd.ndrange(*fem_state.elements.actu.shape, axes=qd.static(eoa(fem_state.elements.actu))):
        force = func_element_internal_force(i_e, i_b, fem_state, fem_info, fem_config)
        i_vs = fem_info.elements.el2v[i_e]
        for i_k in qd.static(range(4)):
            i_v = i_vs[i_k]
            dv = fem_info.substep_dt[None] * fem_info.vertices.mass_inv[i_v] * force[:, i_k]
            qd.atomic_add(fem_state.vertices.vel_next[i_v, i_b], dv)


@qd.kernel(fastcache=True)
def kernel_apply_uniform_force(fem_state: FEMState, fem_info: FEMInfo):
    """Damp the velocities of the vertices over a substep, then add gravity to them, which the damping leaves out."""
    for i_v, i_b in qd.ndrange(*fem_state.vertices.pos.shape, axes=qd.static(eoa(fem_state.vertices.vel))):
        vel = fem_state.vertices.vel_next[i_v, i_b] * fem_info.damping_decay[None]
        fem_state.vertices.vel_next[i_v, i_b] = vel + fem_info.substep_dt[None] * fem_info.gravity[i_b]


@qd.kernel(fastcache=True)
def kernel_compute_pos(fem_state: FEMState, fem_info: FEMInfo, errno: qd.Tensor):
    """
    Advance the positions of the vertices over a substep by their velocity.

    Every solver and coupler path ends a substep here, so the positions it integrates are checked once. The hard
    constraints overwrite some of them afterward, and check their targets (see 'kernel_apply_hard_constraints').
    """
    for i_v, i_b in qd.ndrange(*fem_state.vertices.pos.shape, axes=qd.static(eoa(fem_state.vertices.pos))):
        pos = fem_info.substep_dt[None] * fem_state.vertices.vel_next[i_v, i_b] + fem_state.vertices.pos[i_v, i_b]
        fem_state.vertices.pos_next[i_v, i_b] = pos
        if not qd_isfinite(pos):
            errno[i_b] = errno[i_b] | FEMErrorCode.INVALID_VERTEX_NAN


@qd.kernel(fastcache=True)
def kernel_apply_hard_constraints(
    fem_state: FEMState, links_state: LinksState, vertex_constraints: FEMVertexConstraints, errno: qd.Tensor
):
    """Set the vertices under a hard constraint to their target, at rest."""
    for i_s, i_b in qd.ndrange(
        *vertex_constraints.vertex_idx.shape, axes=qd.static(eoa(vertex_constraints.vertex_idx))
    ):
        if i_s < vertex_constraints.envs_n_constraints[i_b] and not vertex_constraints.is_soft_constraint[i_s, i_b]:
            i_v = vertex_constraints.vertex_idx[i_s, i_b]
            target_pos = func_constraint_target(i_s, i_b, links_state, vertex_constraints)
            fem_state.vertices.pos_next[i_v, i_b] = target_pos
            fem_state.vertices.vel_next[i_v, i_b] = qd.Vector.zero(gs.qd_float, 3)
            if not qd_isfinite(target_pos):
                errno[i_b] = errno[i_b] | FEMErrorCode.INVALID_VERTEX_NAN


@qd.kernel(fastcache=True)
def kernel_apply_soft_constraints(
    fem_state: FEMState, links_state: LinksState, fem_info: FEMInfo, vertex_constraints: FEMVertexConstraints
):
    """Pull the vertices of the soft constraints toward their targets with critically damped springs."""
    for i_s, i_b in qd.ndrange(
        *vertex_constraints.vertex_idx.shape, axes=qd.static(eoa(vertex_constraints.vertex_idx))
    ):
        if i_s < vertex_constraints.envs_n_constraints[i_b] and vertex_constraints.is_soft_constraint[i_s, i_b]:
            i_v = vertex_constraints.vertex_idx[i_s, i_b]
            target_pos = func_constraint_target(i_s, i_b, links_state, vertex_constraints)
            target_vel = qd.Vector.zero(gs.qd_float, 3)
            i_l = vertex_constraints.link_idx[i_s, i_b]
            if i_l >= 0:
                target_vel = links_state.cd_vel[i_l, i_b] + links_state.cd_ang[i_l, i_b].cross(
                    target_pos - links_state.root_COM[i_l, i_b]
                )

            # The spring is integrated by backward Euler, the position moving with the velocity it gives, which keeps
            # it stable whatever its stiffness. The rigid solver has already stepped, so the link places the target at
            # the end of the substep: m (v' - v) = dt (-k (x + dt v' - x_t) - c (v' - v_t)).
            stiffness = vertex_constraints.stiffness[i_s, i_b]
            mass = fem_info.vertices.mass[i_v]
            damping = 2.0 * qd.math.sqrt(stiffness * mass)
            dt = fem_info.substep_dt[None]
            dt_damping_stiffness = dt * damping + dt**2 * stiffness
            vel = fem_state.vertices.vel_next[i_v, i_b]
            pos_error = fem_state.vertices.pos[i_v, i_b] - target_pos
            vel_delta = -(dt_damping_stiffness * vel + dt * stiffness * pos_error - dt * damping * target_vel) / (
                mass + dt_damping_stiffness
            )
            # An accumulation, which autodiff reverses exactly, the kernel reading the velocity it updates
            qd.atomic_add(fem_state.vertices.vel_next[i_v, i_b], vel_delta)


# ------------------------------------------------------------------------------------
# --------------------------------- implicit solver ----------------------------------
# ------------------------------------------------------------------------------------


# The backward pass of the solve reverses the data along with the forces it enters (see '_implicit_solve_grad')
@no_grad
@qd.kernel(fastcache=True)
def kernel_precompute_material_data(fem_state: FEMState, fem_info: FEMInfo, fem_config: qd.template()):
    """Compute the data the materials take from the deformation of the elements at the start of a substep."""
    for i_e, i_b in qd.ndrange(*fem_state.elements.actu.shape, axes=qd.static(eoa(fem_state.elements.actu))):
        _, F = func_element_J_F(i_e, i_b, fem_state.vertices.pos, fem_info)
        func_element_precompute(i_e, i_b, F, fem_state, fem_info, fem_config)


@qd.func
def func_activate_envs(fem_state: FEMState):
    for i_b in range(fem_state.envs_solve.is_active.shape[0]):
        fem_state.envs_solve.is_active[i_b] = True


@qd.kernel(fastcache=True)
def kernel_init_pos_and_inertia(
    fem_state: FEMState, links_state: LinksState, fem_info: FEMInfo, vertex_constraints: FEMVertexConstraints
):
    """
    Start the positions of the vertices at the end of a substep of the implicit solver from those at its start, and
    compute their inertia, a constrained vertex being set to its target.

    Every environment enters the Newton iterations of the substep that follows.
    """
    func_activate_envs(fem_state)
    for i_v, i_b in qd.ndrange(*fem_state.vertices.pos.shape, axes=qd.static(eoa(fem_state.vertices.pos))):
        has_constraints = vertex_constraints.envs_n_constraints[i_b] > 0
        if func_is_vertex_free(i_v, i_b, has_constraints, vertex_constraints):
            fem_state.vertices_energy.inertia[i_v, i_b] = (
                fem_state.vertices.pos[i_v, i_b]
                + fem_state.vertices.vel[i_v, i_b] * fem_info.substep_dt[None]
                + fem_info.gravity[i_b] * fem_info.dt2[None]
            )
            fem_state.vertices.pos_next[i_v, i_b] = fem_state.vertices.pos[i_v, i_b]
    for i_s, i_b in qd.ndrange(
        *vertex_constraints.vertex_idx.shape, axes=qd.static(eoa(vertex_constraints.vertex_idx))
    ):
        if i_s < vertex_constraints.envs_n_constraints[i_b]:
            i_v = vertex_constraints.vertex_idx[i_s, i_b]
            target_pos = func_constraint_target(i_s, i_b, links_state, vertex_constraints)
            fem_state.vertices.pos_next[i_v, i_b] = target_pos
            fem_state.vertices_energy.inertia[i_v, i_b] = target_pos


@qd.func
def func_block_sum(i_t: int, values, n_threads: qd.template()):
    """
    Return the sum of the values over the `n_threads` threads of the block of an environment, the same bitwise on every
    run.

    The subgroup tiles of the block sum first, then every thread adds up the tile sums in order. On CPU, the block is a
    single thread.
    """
    # Threads of the GPU subgroup tile whose sum 'qd_block_sum' computes
    TILE_DIM = qd.static(32)

    totals = values
    if qd.static(n_threads > 1):
        sh_sum = qd.simt.block.SharedArray((values.n, n_threads // TILE_DIM), gs.qd_float)
        for i in qd.static(range(values.n)):
            value_tile = su.qd_block_sum(values[i])
            if i_t % TILE_DIM == 0:
                sh_sum[i, i_t // TILE_DIM] = value_tile
        qd.simt.block.sync()
        totals = qd.Vector.zero(gs.qd_float, values.n)
        for i_tile in range(n_threads // TILE_DIM):
            totals = totals + qd.Vector([sh_sum[i, i_tile] for i in qd.static(range(values.n))])
        qd.simt.block.sync()
    return totals


@qd.func
def func_is_vertex_free(i_v: int, i_b: int, has_constraints: bool, vertex_constraints: FEMVertexConstraints):
    """
    Return whether a vertex moves in the implicit solve, unconstrained.

    The vertex is i_v of the environment i_b, and `has_constraints` whether that environment has any constraint, which
    the caller reads once for all the vertices it checks, so that an environment without constraint reads none of them.
    """
    is_free = True
    if has_constraints:
        is_free = vertex_constraints.vertices_slot[i_v, i_b] < 0
    return is_free


@qd.func
def func_build_coarse(
    fem_state: FEMState,
    fem_info: FEMInfo,
    vertex_constraints: FEMVertexConstraints,
    fem_config: qd.template(),
):
    """
    Compute the coarse matrix of the preconditioner of each entity from the Hessian of a substep (see 'FEMCoarseSolve').

    The Hessian is constant over the substep, so the matrix holds for all its Newton iterations. Column (i_c, i_a) of
    the matrix of an entity, the restriction of the forces of rigid motion i_a of cluster i_c, sums over the elements
    touching the cluster in their order. On GPU, the lanes of a subgroup tile stride them, then add up their sums along
    a fixed tree, and the first lane scales the column by its diagonal entry. The lanes of a tile then invert the matrix
    of an entity (see 'func_coarse_eliminate').
    """
    # Threads of the GPU subgroup tile whose sum 'qd_block_sum' computes
    TILE_DIM = qd.static(32)

    is_serial = qd.static(not fem_config.enable_cooperative_kernels)
    n_lanes = qd.static(1 if is_serial else TILE_DIM)
    n_rows_max = qd.static(6 * fem_config.max_entity_clusters)
    I_3 = qd.Matrix.identity(gs.qd_float, 3)
    # The thread counts are expressions of the array shapes, which a local computed before the loop would turn into a
    # runtime bound stored by a serial task of its own (see 'func_accumulate_vertices').
    for i_flat in range(fem_state.envs_solve.is_active.shape[0] * fem_info.clusters_entity_idx.shape[0] * 6 * n_lanes):
        n_clusters = fem_info.clusters_entity_idx.shape[0]
        i_lane = i_flat % n_lanes
        i_a = i_flat // n_lanes % 6
        i_c = i_flat // (6 * n_lanes) % n_clusters
        i_b = i_flat // (6 * n_lanes * n_clusters)
        i_ent = fem_info.clusters_entity_idx[i_c]
        i_c_start = fem_info.coarse_entities_cluster_start[i_ent]
        n_rows = 6 * (fem_info.coarse_entities_cluster_start[i_ent + 1] - i_c_start)
        column = qd.Vector.zero(gs.qd_float, n_rows_max)
        has_constraints = vertex_constraints.envs_n_constraints[i_b] > 0
        i_ce_start = fem_info.clusters_elements_start[i_c]
        n_cluster_elements = fem_info.clusters_elements_start[i_c + 1] - i_ce_start
        for i_ce_ in range((n_cluster_elements + n_lanes - 1) // n_lanes):
            i_ce = i_ce_ * n_lanes + i_lane
            if i_ce < n_cluster_elements:
                i_e = fem_info.clusters_elements_idx[i_ce_start + i_ce]
                i_vs = fem_info.elements.el2v[i_e]
                # The position of each vertex relative to the center of rotation of its cluster, and the rigid motion
                # i_a of the free vertices of cluster i_c: a translation along axis i_a for i_a < 3, else a rotation
                # about axis i_a - 3.
                d = qd.Matrix.zero(gs.qd_float, 3, 4)
                x = qd.Matrix.zero(gs.qd_float, 3, 4)
                is_free = qd.Vector.zero(gs.qd_int, 4)
                for i_k in qd.static(range(4)):
                    i_v = i_vs[i_k]
                    i_c_k = fem_info.vertices_cluster_idx[i_v]
                    i_cv_start = fem_info.clusters_vertices_start[i_c_k]
                    i_v_ref = fem_info.clusters_vertices_idx[i_cv_start]
                    d[:, i_k] = fem_state.vertices.pos[i_v, i_b] - fem_state.vertices.pos[i_v_ref, i_b]
                    is_free[i_k] = func_is_vertex_free(i_v, i_b, has_constraints, vertex_constraints)
                    if i_c_k == i_c:
                        if is_free[i_k]:
                            for i_x in qd.static(range(3)):
                                if i_a == i_x:
                                    x[i_x, i_k] = 1.0
                                if i_a == 3 + i_x:
                                    x[:, i_k] = I_3[:, i_x].cross(d[:, i_k])
                # Its elastic forces and its inertia from the share of the element in the mass of each of its vertices
                # (see 'kernel_add_elements'), dropped on the constrained vertices.
                S = func_element_mapping_matrix(i_e, fem_info)
                _, dP = func_elastic_product(i_e, i_b, S, x, fem_state, fem_info, fem_config)
                V = fem_info.elements.V[i_e]
                corner_mass = 0.25 * fem_info.elements.mass_scaled[i_e] / fem_info.vol_scale[None]
                forces = V * fem_info.damping_beta_factor[None] * dP @ S.transpose()
                mass_factor = fem_info.damping_alpha_dt[None] + 1.0
                forces = forces + corner_mass * fem_info.dt2_inv[None] * mass_factor * x
                # Their restriction to the rigid motions of the cluster of each vertex
                for i_k in qd.static(range(4)):
                    i_c_k = fem_info.vertices_cluster_idx[i_vs[i_k]]
                    force_k = forces[:, i_k]
                    if not is_free[i_k]:
                        force_k = qd.Vector.zero(gs.qd_float, 3)
                    torque_k = d[:, i_k].cross(force_k)
                    for j_c in qd.static(range(fem_config.max_entity_clusters)):
                        if i_c_k - i_c_start == j_c:
                            for i_x in qd.static(range(3)):
                                column[6 * j_c + i_x] = column[6 * j_c + i_x] + force_k[i_x]
                                column[6 * j_c + 3 + i_x] = column[6 * j_c + 3 + i_x] + torque_k[i_x]
        if qd.static(not is_serial):
            for i_r in qd.static(range(n_rows_max)):
                column[i_r] = su.qd_block_sum(column[i_r])
        if i_lane == 0:
            i_d = 6 * (i_c - i_c_start) + i_a
            diag = gs.qd_float(0.0)
            for i_r in qd.static(range(n_rows_max)):
                if i_r == i_d:
                    diag = column[i_r]
            scale = gs.qd_float(0.0)
            if diag > 0.0:
                scale = 1.0 / qd.sqrt(diag)
            fem_state.coarse_solve.scale[6 * i_c + i_a, i_b] = scale
            i_entry = fem_info.coarse_entities_entry_start[i_ent] + i_d * n_rows
            for i_r in qd.static(range(n_rows_max)):
                if i_r < n_rows:
                    fem_state.coarse_solve.matrix[i_entry + i_r, i_b] = qd.cast(scale * column[i_r], qd.f32)

    # The elimination of the matrix of each entity in the shared memory of a block on GPU, a local matrix on CPU (see
    # 'func_coarse_eliminate')
    qd.loop_config(block_dim=qd.static(n_lanes if n_lanes > 1 else None))
    for i_flat in range(
        fem_state.envs_solve.is_active.shape[0] * (fem_info.coarse_entities_cluster_start.shape[0] - 1) * n_lanes
    ):
        n_entities = fem_info.coarse_entities_cluster_start.shape[0] - 1
        sh_matrix = qd.simt.block.SharedArray((n_rows_max, n_rows_max), qd.f32)
        local_matrix = qd.Matrix.zero(qd.f32, qd.static(n_rows_max if is_serial else 1), n_rows_max)
        i_lane = i_flat % n_lanes
        i_ent = i_flat // n_lanes % n_entities
        i_b = i_flat // (n_lanes * n_entities)
        i_entry_start = fem_info.coarse_entities_entry_start[i_ent]
        i_row_start = 6 * fem_info.coarse_entities_cluster_start[i_ent]
        n_rows = 6 * fem_info.coarse_entities_cluster_start[i_ent + 1] - i_row_start
        if qd.static(is_serial):
            func_coarse_eliminate(
                i_b, i_lane, i_entry_start, i_row_start, n_rows, local_matrix, fem_state, fem_info, fem_config
            )
        else:
            func_coarse_eliminate(
                i_b, i_lane, i_entry_start, i_row_start, n_rows, sh_matrix, fem_state, fem_info, fem_config
            )


@qd.func
def func_coarse_eliminate(
    i_b: int,
    i_lane: int,
    i_entry_start: int,
    i_row_start: int,
    n_rows: int,
    work_matrix: qd.template(),
    fem_state: FEMState,
    fem_info: FEMInfo,
    fem_config: qd.template(),
):
    """
    Invert in place the coarse matrix of an entity of an environment, by the lanes of a block.

    The matrix starts at `i_entry_start` and its rigid motions at `i_row_start`. Each of its columns is stored scaled by
    its diagonal entry, and scaling the other index too makes the diagonal one (see 'FEMCoarseSolve'). The elimination
    runs in `work_matrix`, each lane owning the rows of its rank modulo the number of lanes, and barriers order the
    lanes. A pivot below `coarse_pivot_rtol` marks a rigid motion that the others span or that moves no free vertex: the
    elimination drops it, zeroing its rows and columns of the inverse.
    """
    # Threads of the GPU subgroup tile whose sum 'qd_block_sum' computes
    TILE_DIM = qd.static(32)

    is_serial = qd.static(not fem_config.enable_cooperative_kernels)
    n_lanes = qd.static(1 if is_serial else TILE_DIM)
    n_rows_lane = (n_rows - i_lane + n_lanes - 1) // n_lanes
    for i_q in range(n_rows_lane):
        i_r = i_q * n_lanes + i_lane
        for i_j in range(n_rows):
            value = qd.cast(fem_state.coarse_solve.matrix[i_entry_start + i_r * n_rows + i_j, i_b], gs.qd_float)
            scale_j = fem_state.coarse_solve.scale[i_row_start + i_j, i_b]
            work_matrix[i_r, i_j] = qd.cast(value * scale_j, qd.f32)
    for i_p in range(n_rows):
        if i_p % n_lanes == i_lane:
            pivot_inv = qd.f32(0.0)
            if work_matrix[i_p, i_p] > fem_info.coarse_pivot_rtol[None]:
                pivot_inv = qd.f32(1.0) / work_matrix[i_p, i_p]
            for i_j in range(n_rows):
                work_matrix[i_p, i_j] = pivot_inv * work_matrix[i_p, i_j]
            work_matrix[i_p, i_p] = pivot_inv
        if qd.static(not is_serial):
            qd.simt.block.sync()
        for i_q in range(n_rows_lane):
            i_r = i_q * n_lanes + i_lane
            if i_r != i_p:
                factor = work_matrix[i_r, i_p]
                for i_j in range(n_rows):
                    work_matrix[i_r, i_j] = work_matrix[i_r, i_j] - factor * work_matrix[i_p, i_j]
                work_matrix[i_r, i_p] = -factor * work_matrix[i_p, i_p]
        if qd.static(not is_serial):
            qd.simt.block.sync()
    for i_q in range(n_rows_lane):
        i_r = i_q * n_lanes + i_lane
        for i_j in range(n_rows):
            fem_state.coarse_solve.matrix[i_entry_start + i_r * n_rows + i_j, i_b] = work_matrix[i_r, i_j]


@qd.func
def func_precondition_env(
    i_b: int,
    i_t: int,
    is_active: bool,
    fem_state: FEMState,
    fem_info: FEMInfo,
    vertex_constraints: FEMVertexConstraints,
    n_threads: qd.template(),
):
    """
    Precondition the residual of an environment by the threads of its block, and return the part of its dot product
    with the residual that a thread sums (see 'func_block_sum').

    The preconditioned residual of a vertex is its block Jacobi part, plus for a free vertex the rigid motion of its
    cluster that solves the coarse system (see 'FEMCoarseSolve'). The subgroup tiles of the block take the clusters in
    turn, their lanes adding up the residual of its free vertices restricted to its rigid motions along a fixed tree,
    then each thread solves the rigid motions of a cluster from the coarse matrix of its entity (see
    'func_build_coarse'). The residual of a vertex is read by threads other than the one that wrote it, behind a
    barrier.
    """
    # Threads of the GPU subgroup tile whose sum 'qd_block_sum' computes
    TILE_DIM = qd.static(32)

    is_serial = qd.static(n_threads == 1)
    n_lanes = qd.static(1 if is_serial else TILE_DIM)
    n_tiles = qd.static(n_threads // n_lanes)
    i_tile = i_t // n_lanes
    i_lane = i_t % n_lanes
    n_clusters = fem_info.clusters_entity_idx.shape[0]
    has_constraints = vertex_constraints.envs_n_constraints[i_b] > 0
    if qd.static(not is_serial):
        qd.simt.grid.mem_fence()
        qd.simt.block.sync()
    for i_c_ in range((n_clusters + n_tiles - 1) // n_tiles):
        i_c = i_c_ * n_tiles + i_tile
        rhs = qd.Vector.zero(gs.qd_float, 6)
        if is_active and i_c < n_clusters:
            i_cv_start = fem_info.clusters_vertices_start[i_c]
            n_cluster_vertices = fem_info.clusters_vertices_start[i_c + 1] - i_cv_start
            i_v_ref = fem_info.clusters_vertices_idx[i_cv_start]
            for i_cv_ in range((n_cluster_vertices + n_lanes - 1) // n_lanes):
                i_cv = i_cv_ * n_lanes + i_lane
                if i_cv < n_cluster_vertices:
                    i_v = fem_info.clusters_vertices_idx[i_cv_start + i_cv]
                    if func_is_vertex_free(i_v, i_b, has_constraints, vertex_constraints):
                        r = fem_state.vertices_solve.r[i_v, i_b]
                        d = fem_state.vertices.pos[i_v, i_b] - fem_state.vertices.pos[i_v_ref, i_b]
                        torque = d.cross(r)
                        for i_x in qd.static(range(3)):
                            rhs[i_x] = rhs[i_x] + r[i_x]
                            rhs[3 + i_x] = rhs[3 + i_x] + torque[i_x]
        if qd.static(not is_serial):
            for i_x in qd.static(range(6)):
                rhs[i_x] = su.qd_block_sum(rhs[i_x])
        if is_active and i_c < n_clusters and i_lane == 0:
            fem_state.coarse_solve.rhs[i_c, i_b] = rhs
    if qd.static(not is_serial):
        qd.simt.grid.mem_fence()
        qd.simt.block.sync()
    if is_active:
        for i_c_ in range((n_clusters + n_threads - 1) // n_threads):
            i_c = i_c_ * n_threads + i_t
            if i_c < n_clusters:
                i_ent = fem_info.clusters_entity_idx[i_c]
                i_c_start = fem_info.coarse_entities_cluster_start[i_ent]
                i_c_end = fem_info.coarse_entities_cluster_start[i_ent + 1]
                n_rows = 6 * (i_c_end - i_c_start)
                i_entry = fem_info.coarse_entities_entry_start[i_ent] + 6 * (i_c - i_c_start) * n_rows
                sol = qd.Vector.zero(gs.qd_float, 6)
                for j_c in range(i_c_start, i_c_end):
                    rhs = fem_state.coarse_solve.rhs[j_c, i_b]
                    i_j = 6 * (j_c - i_c_start)
                    for i_x in qd.static(range(6)):
                        rhs_x = fem_state.coarse_solve.scale[6 * j_c + i_x, i_b] * rhs[i_x]
                        for i_a in qd.static(range(6)):
                            entry = qd.cast(
                                fem_state.coarse_solve.matrix[i_entry + i_a * n_rows + i_j + i_x, i_b], gs.qd_float
                            )
                            sol[i_a] = sol[i_a] + entry * rhs_x
                for i_a in qd.static(range(6)):
                    sol[i_a] = fem_state.coarse_solve.scale[6 * i_c + i_a, i_b] * sol[i_a]
                fem_state.coarse_solve.sol[i_c, i_b] = sol
    if qd.static(not is_serial):
        qd.simt.grid.mem_fence()
        qd.simt.block.sync()
    rTz = gs.qd_float(0.0)
    if is_active:
        n_vertices = fem_info.vertices.mass.shape[0]
        for i_v_ in range((n_vertices + n_threads - 1) // n_threads):
            i_v = i_v_ * n_threads + i_t
            if i_v < n_vertices:
                r = fem_state.vertices_solve.r[i_v, i_b]
                z = fem_state.vertices_solve.prec[i_v, i_b] @ r
                i_c = fem_info.vertices_cluster_idx[i_v]
                if i_c >= 0 and func_is_vertex_free(i_v, i_b, has_constraints, vertex_constraints):
                    i_cv_start = fem_info.clusters_vertices_start[i_c]
                    i_v_ref = fem_info.clusters_vertices_idx[i_cv_start]
                    d = fem_state.vertices.pos[i_v, i_b] - fem_state.vertices.pos[i_v_ref, i_b]
                    sol = fem_state.coarse_solve.sol[i_c, i_b]
                    z = z + qd.Vector([sol[0], sol[1], sol[2]]) + qd.Vector([sol[3], sol[4], sol[5]]).cross(d)
                fem_state.vertices_solve.z[i_v, i_b] = z
                rTz = rTz + r.dot(z)
    return rTz


@qd.func
def func_accumulate_vertex(
    i_b: int,
    i_v: int,
    i_lane: int,
    forces: qd.Tensor,
    envs_is_active: qd.Tensor,
    fem_state: FEMState,
    fem_info: FEMInfo,
    vertex_constraints: FEMVertexConstraints,
    fem_config: qd.template(),
    log2_n_lanes: qd.template(),
    gather_Ap: qd.template(),
    gather_internal_force: qd.template(),
    clip_vertex: qd.template(),
):
    """
    Sum the elastic terms of a vertex in a fixed order.

    The internal force alone goes into `forces` for every environment if asked, else Ap alone or the force and diagonal
    block of the active environments. On CPU, the thread of an environment walks the corners of all its elements in
    element order and adds each term to its vertex. On GPU, the 2**`log2_n_lanes` lanes of a vertex stride its corners
    (see 'vertices_corners_idx'), then add up their sums in a fixed order, every lane of a subgroup tile taking part and
    the lanes past the last vertex doing nothing when `clip_vertex` is set.
    """
    is_serial = qd.static(not fem_config.enable_cooperative_kernels)
    n_lanes = qd.static(2**log2_n_lanes)
    n_vertices = fem_info.vertices.mass.shape[0]
    n_elements = fem_info.elements.V.shape[0]
    damping_alpha_dt = fem_info.damping_alpha_dt[None]
    damping_beta_factor = fem_info.damping_beta_factor[None]
    I_3 = qd.Matrix.identity(gs.qd_float, 3)
    vec_elastic = qd.Vector.zero(gs.qd_float, 3)
    mat_elastic = qd.Matrix.zero(gs.qd_float, 3, 3)
    is_active = True
    if qd.static(not gather_internal_force):
        is_active = envs_is_active[i_b]
    if qd.static(clip_vertex):
        is_active = is_active and i_v < n_vertices
    if is_active:
        i_c_start = 0
        n_corners = 4 * n_elements
        if qd.static(is_serial and not gather_internal_force):
            for i_v_ in range(n_vertices):
                if qd.static(gather_Ap):
                    fem_state.vertices_solve.Ap[i_v_, i_b] = qd.Vector.zero(gs.qd_float, 3)
                else:
                    fem_state.vertices_energy.force[i_v_, i_b] = qd.Vector.zero(gs.qd_float, 3)
                    fem_state.vertices_solve.diag3x3[i_v_, i_b] = qd.Matrix.zero(gs.qd_float, 3, 3)
        if qd.static(not is_serial):
            i_c_start = fem_info.vertices_corners_start[i_v]
            n_corners = fem_info.vertices_corners_start[i_v + 1] - i_c_start

        # On CPU the corners of an element follow each other, sharing its mapping matrix, stress or force
        S = qd.Matrix.zero(gs.qd_float, 4, 3)
        stress = qd.Matrix.zero(gs.qd_float, 3, 3)
        force_internal = qd.Matrix.zero(gs.qd_float, 3, 4)
        for i_c_ in range((n_corners + n_lanes - 1) // n_lanes):
            i_c = i_c_ * n_lanes + i_lane
            if i_c < n_corners:
                i_corner = i_c
                if qd.static(not is_serial):
                    i_corner = fem_info.vertices_corners_idx[i_c_start + i_c]
                i_e = i_corner // 4
                i_k = i_corner % 4
                V = fem_info.elements.V[i_e]
                i_vs = fem_info.elements.el2v[i_e]
                is_new_element = True
                if qd.static(is_serial):
                    is_new_element = i_k == 0
                if is_new_element:
                    if qd.static(gather_internal_force):
                        force_internal = func_element_internal_force(i_e, i_b, fem_state, fem_info, fem_config)
                    else:
                        S = func_element_mapping_matrix(i_e, fem_info)
                    if qd.static(gather_Ap and is_serial):
                        p = qd.Matrix.cols([fem_state.vertices_solve.p[i_vs[i_j], i_b] for i_j in qd.static(range(4))])
                        _, dP = func_elastic_product(i_e, i_b, S, p, fem_state, fem_info, fem_config)
                        stress = V * damping_beta_factor * dP
                # On GPU, each element computes its stress once for all its corners (see 'FEMElementsEnergy')
                if qd.static(not gather_internal_force and not (gather_Ap and is_serial)):
                    stress = fem_state.elements_energy.stress[i_e, i_b]
                S_k = S[i_k, :]

                vec_corner = qd.Vector.zero(gs.qd_float, 3)
                mat_corner = qd.Matrix.zero(gs.qd_float, 3, 3)
                if qd.static(gather_internal_force):
                    vec_corner = force_internal[:, i_k]
                else:
                    vec_corner = stress @ S_k
                if qd.static(not gather_Ap and not gather_internal_force):
                    # Column j of the block is the force a unit displacement of the vertex along axis j induces on it
                    corner_k = qd.Matrix.identity(gs.qd_float, 4)[i_k, :]
                    for j in range(3):
                        _, dP = func_elastic_product(
                            i_e, i_b, S, I_3[:, j].outer_product(corner_k), fem_state, fem_info, fem_config
                        )
                        mat_corner = mat_corner + (V * damping_beta_factor * dP @ S_k).outer_product(I_3[:, j])

                if qd.static(is_serial):
                    i_v_k = i_vs[i_k]
                    if qd.static(gather_internal_force):
                        for j in qd.static(range(3)):
                            forces[i_b, i_v_k, j] = forces[i_b, i_v_k, j] + vec_corner[j]
                    elif qd.static(gather_Ap):
                        fem_state.vertices_solve.Ap[i_v_k, i_b] = fem_state.vertices_solve.Ap[i_v_k, i_b] + vec_corner
                    else:
                        force = fem_state.vertices_energy.force[i_v_k, i_b] + vec_corner
                        fem_state.vertices_energy.force[i_v_k, i_b] = force
                        diag3x3 = fem_state.vertices_solve.diag3x3[i_v_k, i_b] + mat_corner
                        fem_state.vertices_solve.diag3x3[i_v_k, i_b] = diag3x3
                else:
                    vec_elastic = vec_elastic + vec_corner
                    mat_elastic = mat_elastic + mat_corner
    if qd.static(not is_serial):
        for i in qd.static(range(3)):
            vec_elastic[i] = su.qd_block_sum(vec_elastic[i], log2_n_lanes)
            if qd.static(not gather_Ap and not gather_internal_force):
                for j in qd.static(range(3)):
                    mat_elastic[i, j] = su.qd_block_sum(mat_elastic[i, j], log2_n_lanes)

    if qd.static(gather_internal_force):
        # The first lane of a vertex writes its sum, which the walk over the corners adds up directly on CPU
        if qd.static(not is_serial):
            if i_lane == 0:
                for j in qd.static(range(3)):
                    forces[i_b, i_v, j] = vec_elastic[j]
    elif is_active:
        # The inertia terms of the vertices this thread owns add to their elastic sums
        n_owned = 1 if i_lane == 0 else 0
        if qd.static(is_serial):
            n_owned = n_vertices
        has_constraints = vertex_constraints.envs_n_constraints[i_b] > 0
        for i_o in range(n_owned):
            i_v_o = i_v
            if qd.static(is_serial):
                i_v_o = i_o
                vec_elastic = fem_state.vertices_energy.force[i_v_o, i_b]
                mat_elastic = fem_state.vertices_solve.diag3x3[i_v_o, i_b]
                if qd.static(gather_Ap):
                    vec_elastic = fem_state.vertices_solve.Ap[i_v_o, i_b]
            # A constrained vertex follows its target, the elastic forces on it dropped
            if not func_is_vertex_free(i_v_o, i_b, has_constraints, vertex_constraints):
                vec_elastic = qd.Vector.zero(gs.qd_float, 3)
                mat_elastic = qd.Matrix.zero(gs.qd_float, 3, 3)
            mass_over_dt2 = fem_info.vertices.mass_over_dt2[i_v_o]
            if qd.static(gather_Ap):
                p = fem_state.vertices_solve.p[i_v_o, i_b]
                fem_state.vertices_solve.Ap[i_v_o, i_b] = mass_over_dt2 * (damping_alpha_dt + 1.0) * p + vec_elastic
            else:
                force_inertia = func_vertex_inertia_force(
                    i_v_o, i_b, has_constraints, fem_state, fem_info, vertex_constraints
                )
                fem_state.vertices_energy.force[i_v_o, i_b] = force_inertia + vec_elastic
                diag3x3 = mat_elastic + mass_over_dt2 * (damping_alpha_dt + 1.0) * I_3
                fem_state.vertices_solve.diag3x3[i_v_o, i_b] = diag3x3
                # 3-by-3 block Jacobi preconditioner
                fem_state.vertices_solve.prec[i_v_o, i_b] = diag3x3.inverse()


@qd.func
def func_accumulate_vertices(
    forces: qd.Tensor,
    envs_is_active: qd.Tensor,
    pcg_counter: qd.Tensor,
    fem_state: FEMState,
    fem_info: FEMInfo,
    vertex_constraints: FEMVertexConstraints,
    fem_config: qd.template(),
    gather_Ap: qd.template(),
    gather_internal_force: qd.template(),
    update_loop_flag: qd.template(),
):
    """Sum the elastic terms of every vertex of every environment (see 'func_accumulate_vertex')."""
    # Lanes of a GPU subgroup tile that gather together the element corners of a vertex (see 'vertices_corners_idx')
    LOG2_CORNER_LANES = qd.static(2)

    is_serial = qd.static(not fem_config.enable_cooperative_kernels)
    log2_n_lanes = qd.static(0 if is_serial else LOG2_CORNER_LANES)
    n_lanes = qd.static(2**log2_n_lanes)
    if qd.static(gather_Ap and not is_serial):
        for i_e, i_b in qd.ndrange(
            *fem_state.elements_energy.stress.shape, axes=qd.static(eoa(fem_state.elements_energy.stress))
        ):
            # A PCG iteration starts here: its loop flag drops until an environment iterates on (see
            # 'kernel_batch_solve')
            if qd.static(update_loop_flag):
                if i_e == 0 and i_b == 0:
                    pcg_counter[()] = 0
                    fem_state.solve_loops.n_pcg_left[()] = fem_state.solve_loops.n_pcg_left[()] - 1
            # The stress along the search direction (see 'FEMElementsEnergy')
            if envs_is_active[i_b]:
                S = func_element_mapping_matrix(i_e, fem_info)
                i_vs = fem_info.elements.el2v[i_e]
                p = qd.Matrix.cols([fem_state.vertices_solve.p[i_vs[i_k], i_b] for i_k in qd.static(range(4))])
                _, dP = func_elastic_product(i_e, i_b, S, p, fem_state, fem_info, fem_config)
                V = fem_info.elements.V[i_e]
                fem_state.elements_energy.stress[i_e, i_b] = V * fem_info.damping_beta_factor[None] * dP
    # The thread count is an expression of the array shapes, which fields make a compile-time constant. A local computed
    # before the loop would be stored by a serial task of its own and read back as a runtime bound.
    for i_flat in range(
        fem_state.vertices.pos.shape[1] * (1 if is_serial else fem_info.vertices.mass.shape[0] * n_lanes)
    ):
        n_vertices = fem_info.vertices.mass.shape[0]
        n_env_threads = 1
        if qd.static(not is_serial):
            n_env_threads = n_vertices * n_lanes
        # A PCG iteration starts here on CPU, its first loop on GPU being over the elements
        if qd.static(gather_Ap and update_loop_flag and is_serial):
            if i_flat == 0:
                pcg_counter[()] = 0
                fem_state.solve_loops.n_pcg_left[()] = fem_state.solve_loops.n_pcg_left[()] - 1
        i_b = i_flat // n_env_threads
        i_lane = i_flat % n_lanes
        i_v = i_flat // n_lanes % n_vertices
        func_accumulate_vertex(
            i_b,
            i_v,
            i_lane,
            forces,
            envs_is_active,
            fem_state,
            fem_info,
            vertex_constraints,
            fem_config,
            log2_n_lanes,
            gather_Ap,
            gather_internal_force,
            clip_vertex=False,
        )


@qd.func
def func_init_pcg(
    pcg_counter: qd.Tensor,
    fem_state: FEMState,
    fem_info: FEMInfo,
    vertex_constraints: FEMVertexConstraints,
    fem_config: qd.template(),
    is_adjoint: qd.template(),
):
    """
    Compute the first residual of the PCG solve of each environment, by the threads of its block (see 'func_block_sum').

    It is the forces at the end of the substep, or for the adjoint solve the adjoint of the positions there (see
    'FEMSolver._implicit_solve_grad' for that solve).
    """
    # Threads of the GPU block that sums the terms of an environment over its vertices and elements in a fixed order
    BLOCK_DIM = qd.static(256)

    n_threads = qd.static(BLOCK_DIM if fem_config.enable_cooperative_kernels else 1)

    pcg_counter[()] = 0
    qd.loop_config(block_dim=qd.static(n_threads if n_threads > 1 else None))
    for i_flat in range(fem_state.envs_solve.is_active.shape[0] * n_threads):
        i_b = i_flat // n_threads
        i_t = i_flat % n_threads
        n_vertices = fem_info.vertices.mass.shape[0]
        is_active = fem_state.envs_solve.is_active[i_b]
        has_constraints = vertex_constraints.envs_n_constraints[i_b] > 0
        rTr = gs.qd_float(0.0)
        if is_active:
            for i_v_ in range((n_vertices + n_threads - 1) // n_threads):
                i_v = i_v_ * n_threads + i_t
                if i_v < n_vertices:
                    r = fem_state.vertices_energy.force[i_v, i_b]
                    if qd.static(is_adjoint):
                        r = fem_state.vertices.pos_next.grad[i_v, i_b]
                    # A constrained vertex stays out of the solve, its search direction staying zero
                    if not func_is_vertex_free(i_v, i_b, has_constraints, vertex_constraints):
                        r = qd.Vector.zero(gs.qd_float, 3)
                    fem_state.vertices_solve.x[i_v, i_b] = qd.Vector.zero(gs.qd_float, 3)
                    fem_state.vertices_solve.r[i_v, i_b] = r
                    rTr = rTr + r.dot(r)
        rTz = func_precondition_env(i_b, i_t, is_active, fem_state, fem_info, vertex_constraints, n_threads)
        if is_active:
            for i_v_ in range((n_vertices + n_threads - 1) // n_threads):
                i_v = i_v_ * n_threads + i_t
                if i_v < n_vertices:
                    fem_state.vertices_solve.p[i_v, i_b] = fem_state.vertices_solve.z[i_v, i_b]
        rTr, rTz = func_block_sum(i_t, values=qd.Vector([rTr, rTz]), n_threads=n_threads)
        if i_t == 0:
            if is_active:
                fem_state.envs_solve.rTr[i_b] = rTr
                fem_state.envs_solve.rTz[i_b] = rTz
            rTr_threshold = fem_info.pcg_threshold[None] * rTr
            fem_state.envs_solve.rTr_threshold[i_b] = rTr_threshold
            is_pcg_active = is_active and rTr > rTr_threshold
            fem_state.envs_solve.is_pcg_active[i_b] = is_pcg_active
            if is_pcg_active:
                pcg_counter[()] = 1
    fem_state.solve_loops.n_pcg_left[()] = fem_info.n_pcg_iterations[None]


@qd.func
def func_update_pcg_env(
    i_b: int,
    i_t: int,
    pcg_counter: qd.Tensor,
    fem_state: FEMState,
    fem_info: FEMInfo,
    vertex_constraints: FEMVertexConstraints,
    n_threads: qd.template(),
    update_loop_flag: qd.template(),
):
    """Run the step of a PCG iteration of an environment past its Ap on its block of threads (see 'func_block_sum')."""
    n_vertices = fem_info.vertices.mass.shape[0]
    is_active = fem_state.envs_solve.is_pcg_active[i_b]
    rTz = fem_state.envs_solve.rTz[i_b]

    # pTAp, then x, r, z and the new rTr, rTz
    pTAp = gs.qd_float(0.0)
    if is_active:
        for i_v_ in range((n_vertices + n_threads - 1) // n_threads):
            i_v = i_v_ * n_threads + i_t
            if i_v < n_vertices:
                pTAp = pTAp + fem_state.vertices_solve.p[i_v, i_b].dot(fem_state.vertices_solve.Ap[i_v, i_b])
    pTAp = func_block_sum(i_t, values=qd.Vector([pTAp]), n_threads=n_threads)[0]
    alpha = gs.qd_float(0.0)
    rTr_new = gs.qd_float(0.0)
    if is_active:
        alpha = rTz / pTAp
        for i_v_ in range((n_vertices + n_threads - 1) // n_threads):
            i_v = i_v_ * n_threads + i_t
            if i_v < n_vertices:
                p = fem_state.vertices_solve.p[i_v, i_b]
                r = fem_state.vertices_solve.r[i_v, i_b] - alpha * fem_state.vertices_solve.Ap[i_v, i_b]
                fem_state.vertices_solve.x[i_v, i_b] = fem_state.vertices_solve.x[i_v, i_b] + alpha * p
                fem_state.vertices_solve.r[i_v, i_b] = r
                rTr_new = rTr_new + r.dot(r)
    rTz_new = func_precondition_env(i_b, i_t, is_active, fem_state, fem_info, vertex_constraints, n_threads)
    rTr_new, rTz_new = func_block_sum(i_t, values=qd.Vector([rTr_new, rTz_new]), n_threads=n_threads)

    # Convergence check, then the new search direction
    is_still_active = is_active and rTr_new > fem_state.envs_solve.rTr_threshold[i_b]
    beta = gs.qd_float(0.0)
    if is_still_active:
        beta = rTz_new / rTz
        for i_v_ in range((n_vertices + n_threads - 1) // n_threads):
            i_v = i_v_ * n_threads + i_t
            if i_v < n_vertices:
                fem_state.vertices_solve.p[i_v, i_b] = (
                    fem_state.vertices_solve.z[i_v, i_b] + beta * fem_state.vertices_solve.p[i_v, i_b]
                )
    if i_t == 0:
        if is_still_active:
            fem_state.envs_solve.rTr[i_b] = rTr_new
            fem_state.envs_solve.rTz[i_b] = rTz_new
            if qd.static(update_loop_flag):
                if fem_state.solve_loops.n_pcg_left[()] > 0:
                    pcg_counter[()] = 1
        fem_state.envs_solve.is_pcg_active[i_b] = is_still_active


@qd.func
def func_pcg_iter(
    pcg_counter: qd.Tensor,
    fem_state: FEMState,
    fem_info: FEMInfo,
    vertex_constraints: FEMVertexConstraints,
    fem_config: qd.template(),
):
    # See BLOCK_DIM in 'func_init_pcg'
    BLOCK_DIM = qd.static(256)

    n_threads = qd.static(BLOCK_DIM if fem_config.enable_cooperative_kernels else 1)

    func_accumulate_vertices(
        forces=None,
        envs_is_active=fem_state.envs_solve.is_pcg_active,
        pcg_counter=pcg_counter,
        fem_state=fem_state,
        fem_info=fem_info,
        vertex_constraints=vertex_constraints,
        fem_config=fem_config,
        gather_Ap=True,
        gather_internal_force=False,
        update_loop_flag=True,
    )
    qd.loop_config(block_dim=qd.static(n_threads if n_threads > 1 else None))
    for i_flat in range(fem_state.envs_solve.is_active.shape[0] * n_threads):
        i_b = i_flat // n_threads
        i_t = i_flat % n_threads
        func_update_pcg_env(
            i_b, i_t, pcg_counter, fem_state, fem_info, vertex_constraints, n_threads, update_loop_flag=True
        )


@qd.func
def func_pcg_solve_per_env_block(
    pcg_counter: qd.Tensor,
    fem_state: FEMState,
    fem_info: FEMInfo,
    vertex_constraints: FEMVertexConstraints,
    fem_config: qd.template(),
):
    """
    Run the whole PCG solve of an environment by one block of threads.

    The threads synchronize between the steps of an iteration within the kernel (see 'is_pcg_solved_per_env_block').
    """
    # See BLOCK_DIM in 'func_init_pcg'
    BLOCK_DIM = qd.static(256)
    # See LOG2_CORNER_LANES in 'func_accumulate_vertices'
    LOG2_CORNER_LANES = qd.static(2)

    n_lanes = qd.static(2**LOG2_CORNER_LANES)
    n_vertices = fem_info.vertices.mass.shape[0]
    qd.loop_config(block_dim=BLOCK_DIM)
    for i_flat in range(fem_state.envs_solve.is_active.shape[0] * BLOCK_DIM):
        i_b = i_flat // BLOCK_DIM
        i_t = i_flat % BLOCK_DIM
        for i_iter in range(fem_info.n_pcg_iterations[None]):
            # Every thread of the block reads the flag behind the barrier that follows its last write
            if not fem_state.envs_solve.is_pcg_active[i_b]:
                break
            for i_e_ in range((fem_info.elements.V.shape[0] + BLOCK_DIM - 1) // BLOCK_DIM):
                i_e = i_e_ * BLOCK_DIM + i_t
                if i_e < fem_info.elements.V.shape[0]:
                    S = func_element_mapping_matrix(i_e, fem_info)
                    i_vs = fem_info.elements.el2v[i_e]
                    p = qd.Matrix.cols([fem_state.vertices_solve.p[i_vs[i_k], i_b] for i_k in qd.static(range(4))])
                    _, dP = func_elastic_product(i_e, i_b, S, p, fem_state, fem_info, fem_config)
                    V = fem_info.elements.V[i_e]
                    fem_state.elements_energy.stress[i_e, i_b] = V * fem_info.damping_beta_factor[None] * dP
            qd.simt.grid.mem_fence()
            qd.simt.block.sync()
            for i_item_ in range((n_vertices * n_lanes + BLOCK_DIM - 1) // BLOCK_DIM):
                i_item = i_item_ * BLOCK_DIM + i_t
                func_accumulate_vertex(
                    i_b=i_b,
                    i_v=i_item // n_lanes,
                    i_lane=i_item % n_lanes,
                    forces=None,
                    envs_is_active=fem_state.envs_solve.is_pcg_active,
                    fem_state=fem_state,
                    fem_info=fem_info,
                    vertex_constraints=vertex_constraints,
                    fem_config=fem_config,
                    log2_n_lanes=LOG2_CORNER_LANES,
                    gather_Ap=True,
                    gather_internal_force=False,
                    clip_vertex=True,
                )
            qd.simt.grid.mem_fence()
            qd.simt.block.sync()
            func_update_pcg_env(
                i_b,
                i_t,
                pcg_counter,
                fem_state,
                fem_info,
                vertex_constraints,
                n_threads=BLOCK_DIM,
                update_loop_flag=False,
            )
            qd.simt.grid.mem_fence()
            qd.simt.block.sync()


@qd.func
def func_pcg_solve(
    pcg_counter: qd.Tensor,
    fem_state: FEMState,
    fem_info: FEMInfo,
    vertex_constraints: FEMVertexConstraints,
    fem_config: qd.template(),
):
    if qd.static(fem_config.is_pcg_solved_per_env_block):
        func_pcg_solve_per_env_block(pcg_counter, fem_state, fem_info, vertex_constraints, fem_config)
    else:
        while qd.graph.do_while(pcg_counter):
            func_pcg_iter(pcg_counter, fem_state, fem_info, vertex_constraints, fem_config)


@qd.func
def func_env_energy_part(
    i_b: int,
    i_t: int,
    fem_state: FEMState,
    fem_info: FEMInfo,
    fem_config: qd.template(),
    n_threads: qd.template(),
    add_damping_alpha: qd.template(),
):
    """
    Return the part of the incremental energy of an environment that a thread of its block sums (see 'func_block_sum').

    The energy sums the inertia of the vertices, plus their Rayleigh mass damping if `add_damping_alpha` holds, then the
    energy of the elements as computed last (see 'func_linesearch_iter').
    """
    n_vertices = fem_info.vertices.mass.shape[0]
    n_elements = fem_info.elements.V.shape[0]
    energy = gs.qd_float(0.0)
    for i_v_ in range((n_vertices + n_threads - 1) // n_threads):
        i_v = i_v_ * n_threads + i_t
        if i_v < n_vertices:
            mass_over_dt2 = fem_info.vertices.mass_over_dt2[i_v]
            pos = fem_state.vertices.pos_next[i_v, i_b]
            diff = pos - fem_state.vertices_energy.inertia[i_v, i_b]
            energy = energy + 0.5 * mass_over_dt2 * diff.dot(diff)
            if qd.static(add_damping_alpha and fem_config.has_damping_alpha):
                diff = pos - fem_state.vertices.pos[i_v, i_b]
                energy = energy + 0.5 * mass_over_dt2 * diff.dot(diff) * fem_info.damping_alpha_dt[None]
    for i_e_ in range((n_elements + n_threads - 1) // n_threads):
        i_e = i_e_ * n_threads + i_t
        if i_e < n_elements:
            energy = energy + fem_state.elements_energy.energy[i_e, i_b] * fem_info.elements.V[i_e]
    return energy


@qd.func
def func_init_linesearch(
    linesearch_counter: qd.Tensor, fem_state: FEMState, fem_info: FEMInfo, fem_config: qd.template()
):
    # See BLOCK_DIM in 'func_init_pcg'
    BLOCK_DIM = qd.static(256)

    n_threads = qd.static(BLOCK_DIM if fem_config.enable_cooperative_kernels else 1)
    linesearch_counter[()] = 0
    n_vertices = fem_info.vertices.mass.shape[0]
    qd.loop_config(block_dim=qd.static(n_threads if n_threads > 1 else None))
    for i_flat in range(fem_state.envs_solve.is_active.shape[0] * n_threads):
        i_b = i_flat // n_threads
        i_t = i_flat % n_threads
        is_active = fem_state.envs_solve.is_active[i_b]
        prev_energy = gs.qd_float(0.0)
        m = gs.qd_float(0.0)
        if is_active:
            for i_v_ in range((n_vertices + n_threads - 1) // n_threads):
                i_v = i_v_ * n_threads + i_t
                if i_v < n_vertices:
                    fem_state.vertices_solve.x_prev[i_v, i_b] = fem_state.vertices.pos_next[i_v, i_b]
                    m = m - fem_state.vertices_solve.x[i_v, i_b].dot(fem_state.vertices_energy.force[i_v, i_b])
            prev_energy = func_env_energy_part(
                i_b, i_t, fem_state, fem_info, fem_config, n_threads, add_damping_alpha=False
            )
        prev_energy, m = func_block_sum(i_t, values=qd.Vector([prev_energy, m]), n_threads=n_threads)
        if i_t == 0:
            fem_state.envs_solve.is_linesearch_active[i_b] = is_active
            if is_active:
                linesearch_counter[()] = 1
                fem_state.envs_solve.prev_energy[i_b] = prev_energy
                fem_state.envs_solve.step_size[i_b] = 1.0
                fem_state.envs_solve.m[i_b] = m
    fem_state.solve_loops.n_linesearch_left[()] = fem_info.n_linesearch_iterations[None]


@qd.func
def func_linesearch_iter(
    linesearch_counter: qd.Tensor, fem_state: FEMState, fem_info: FEMInfo, fem_config: qd.template()
):
    # See BLOCK_DIM in 'func_init_pcg'
    BLOCK_DIM = qd.static(256)

    n_threads = qd.static(BLOCK_DIM if fem_config.enable_cooperative_kernels else 1)

    _B = fem_state.envs_solve.is_active.shape[0]
    n_vertices = fem_info.vertices.mass.shape[0]
    for i_v, i_b in qd.ndrange(*fem_state.vertices_solve.x.shape, axes=qd.static(eoa(fem_state.vertices_solve.x))):
        # A line search iteration starts here: its loop flag drops until an environment iterates on
        if i_b == 0 and i_v == 0:
            linesearch_counter[()] = 0
            fem_state.solve_loops.n_linesearch_left[()] = fem_state.solve_loops.n_linesearch_left[()] - 1
        if fem_state.envs_solve.is_linesearch_active[i_b]:
            fem_state.vertices.pos_next[i_v, i_b] = (
                fem_state.vertices_solve.x_prev[i_v, i_b]
                + fem_state.envs_solve.step_size[i_b] * fem_state.vertices_solve.x[i_v, i_b]
            )

    # The energy of each element of the environments the line search runs, its linearized damping included
    for i_e, i_b in qd.ndrange(
        *fem_state.elements_energy.energy.shape, axes=qd.static(eoa(fem_state.elements_energy.energy))
    ):
        if fem_state.envs_solve.is_linesearch_active[i_b]:
            J, F = func_element_J_F(i_e, i_b, fem_state.vertices.pos_next, fem_info)
            actu = fem_state.elements.actu[i_e, i_b]
            energy = func_element_energy(i_e, i_b, J, F, actu, fem_state, fem_info, fem_config)
            if qd.static(fem_config.has_damping_beta):
                S = func_element_mapping_matrix(i_e, fem_info)
                x = func_element_displacement(i_e, i_b, fem_state, fem_info)
                dF, dP = func_elastic_product(i_e, i_b, S, x, fem_state, fem_info, fem_config)
                energy = energy + 0.5 * fem_info.damping_beta_over_dt[None] * (dF * dP).sum()
            fem_state.elements_energy.energy[i_e, i_b] = energy

    qd.loop_config(block_dim=qd.static(n_threads if n_threads > 1 else None))
    for i_flat in range(_B * n_threads):
        i_b = i_flat // n_threads
        i_t = i_flat % n_threads
        is_active = fem_state.envs_solve.is_linesearch_active[i_b]
        energy = gs.qd_float(0.0)
        if is_active:
            energy = func_env_energy_part(i_b, i_t, fem_state, fem_info, fem_config, n_threads, add_damping_alpha=True)
        energy = func_block_sum(i_t, values=qd.Vector([energy]), n_threads=n_threads)[0]

        step_size = fem_state.envs_solve.step_size[i_b]
        is_rejected = is_active and (
            energy
            > fem_state.envs_solve.prev_energy[i_b]
            + fem_info.linesearch_c[None] * step_size * fem_state.envs_solve.m[i_b]
        )
        # A search that runs out of trials takes no step, so that the positions always follow the stored step
        is_exhausted = is_rejected and fem_state.solve_loops.n_linesearch_left[()] <= 0
        if is_exhausted:
            for i_v_ in range((n_vertices + n_threads - 1) // n_threads):
                i_v = i_v_ * n_threads + i_t
                if i_v < n_vertices:
                    fem_state.vertices.pos_next[i_v, i_b] = fem_state.vertices_solve.x_prev[i_v, i_b]
        if i_t == 0 and is_active:
            fem_state.envs_solve.is_linesearch_active[i_b] = is_rejected and not is_exhausted
            if is_exhausted:
                fem_state.envs_solve.step_size[i_b] = 0.0
            elif is_rejected:
                linesearch_counter[()] = 1
                fem_state.envs_solve.step_size[i_b] = step_size * fem_info.linesearch_tau[None]


@qd.func
def func_skip_linesearch(fem_state: FEMState, fem_info: FEMInfo):
    for i_v, i_b in qd.ndrange(*fem_state.vertices_solve.x.shape, axes=qd.static(eoa(fem_state.vertices_solve.x))):
        if fem_state.envs_solve.is_active[i_b]:
            if i_v == 0:
                fem_state.envs_solve.step_size[i_b] = 1.0
            fem_state.vertices.pos_next[i_v, i_b] = (
                fem_state.vertices.pos_next[i_v, i_b] + fem_state.vertices_solve.x[i_v, i_b]
            )


@qd.func
def func_end_newton(newton_counter: qd.Tensor, fem_state: FEMState, fem_info: FEMInfo, fem_config: qd.template()):
    """
    End a Newton iteration of each environment, which stops iterating once its vertices moved by less than the threshold
    over the iteration, in root mean square.

    The loop flag drops until an environment iterates on with iterations left.
    """
    # See BLOCK_DIM in 'func_init_pcg'
    BLOCK_DIM = qd.static(256)

    n_threads = qd.static(BLOCK_DIM if fem_config.enable_cooperative_kernels else 1)
    n_vertices = fem_info.vertices.mass.shape[0]
    fem_state.solve_loops.n_newton_left[()] = fem_state.solve_loops.n_newton_left[()] - 1
    newton_counter[()] = 0
    qd.loop_config(block_dim=qd.static(n_threads if n_threads > 1 else None))
    for i_flat in range(fem_state.envs_solve.is_active.shape[0] * n_threads):
        i_b = i_flat // n_threads
        i_t = i_flat % n_threads
        is_active = fem_state.envs_solve.is_active[i_b]
        dx_sqr = gs.qd_float(0.0)
        if is_active:
            step_size = fem_state.envs_solve.step_size[i_b]
            for i_v_ in range((n_vertices + n_threads - 1) // n_threads):
                i_v = i_v_ * n_threads + i_t
                if i_v < n_vertices:
                    dx_sqr = dx_sqr + (step_size * fem_state.vertices_solve.x[i_v, i_b]).norm_sqr()
        dx_sqr = func_block_sum(i_t, values=qd.Vector([dx_sqr]), n_threads=n_threads)[0]
        if i_t == 0 and is_active:
            if dx_sqr < n_vertices * fem_info.newton_dx_threshold[None] ** 2:
                fem_state.envs_solve.is_active[i_b] = False
            elif fem_state.solve_loops.n_newton_left[()] > 0:
                newton_counter[()] = 1


@qd.func
def func_newton_head(
    pcg_counter: qd.Tensor,
    fem_state: FEMState,
    fem_info: FEMInfo,
    vertex_constraints: FEMVertexConstraints,
    fem_config: qd.template(),
):
    """Compute the energy gradient, force and preconditioner of a Newton iteration, then the first PCG residual."""
    for i_e, i_b in qd.ndrange(
        *fem_state.elements_energy.energy.shape, axes=qd.static(eoa(fem_state.elements_energy.energy))
    ):
        if fem_state.envs_solve.is_active[i_b]:
            energy, gradient = func_element_energy_gradient_damped(i_e, i_b, fem_state, fem_info, fem_config)
            fem_state.elements_energy.energy[i_e, i_b] = energy
            fem_state.elements_energy.stress[i_e, i_b] = -fem_info.elements.V[i_e] * gradient
    func_accumulate_vertices(
        forces=None,
        envs_is_active=fem_state.envs_solve.is_active,
        pcg_counter=pcg_counter,
        fem_state=fem_state,
        fem_info=fem_info,
        vertex_constraints=vertex_constraints,
        fem_config=fem_config,
        gather_Ap=False,
        gather_internal_force=False,
        update_loop_flag=False,
    )
    func_init_pcg(pcg_counter, fem_state, fem_info, vertex_constraints, fem_config, is_adjoint=False)


@qd.kernel(graph=True, fastcache=True)
def kernel_batch_solve(
    newton_counter: qd.types.ndarray(qd.i32, ndim=0),
    pcg_counter: qd.types.ndarray(qd.i32, ndim=0),
    linesearch_counter: qd.types.ndarray(qd.i32, ndim=0),
    fem_state: FEMState,
    fem_info: FEMInfo,
    vertex_constraints: FEMVertexConstraints,
    fem_config: qd.template(),
):
    """
    Run the Newton iterations of a substep of the implicit solver.

    Each iteration solves its linear system by PCG, then searches its step along the solution.
    """
    func_build_coarse(fem_state, fem_info, vertex_constraints, fem_config)
    newton_counter[()] = 1
    fem_state.solve_loops.n_newton_left[()] = fem_info.n_newton_iterations[None]
    # FIXME: quadrants#1083 - the statements before a graph do_while run again on every pass when the loop body
    # opens with bare statements, so each loop body opens with a for loop.
    while qd.graph.do_while(newton_counter):
        func_newton_head(pcg_counter, fem_state, fem_info, vertex_constraints, fem_config)
        func_pcg_solve(pcg_counter, fem_state, fem_info, vertex_constraints, fem_config)
        if qd.static(fem_config.has_linesearch):
            func_init_linesearch(linesearch_counter, fem_state, fem_info, fem_config)
            while qd.graph.do_while(linesearch_counter):
                func_linesearch_iter(linesearch_counter, fem_state, fem_info, fem_config)
        else:
            func_skip_linesearch(fem_state, fem_info)
        func_end_newton(newton_counter, fem_state, fem_info, fem_config)


@qd.func
def func_seed_force_adjoint(fem_state: FEMState):
    for i_v, i_b in qd.ndrange(
        *fem_state.vertices_energy.force.shape, axes=qd.static(eoa(fem_state.vertices_energy.force))
    ):
        fem_state.vertices_energy.force.grad[i_v, i_b] = fem_state.vertices_solve.x[i_v, i_b]


@qd.kernel(graph=True, fastcache=True)
def kernel_solve_adjoint(
    pcg_counter: qd.types.ndarray(qd.i32, ndim=0),
    fem_state: FEMState,
    fem_info: FEMInfo,
    vertex_constraints: FEMVertexConstraints,
    fem_config: qd.template(),
):
    """Run the adjoint solve of a substep, which seeds the adjoint of the forces (see '_implicit_solve_grad')."""
    func_activate_envs(fem_state)
    func_build_coarse(fem_state, fem_info, vertex_constraints, fem_config)
    func_init_pcg(pcg_counter, fem_state, fem_info, vertex_constraints, fem_config, is_adjoint=True)
    func_pcg_solve(pcg_counter, fem_state, fem_info, vertex_constraints, fem_config)
    func_seed_force_adjoint(fem_state)


@qd.kernel(fastcache=True)
def kernel_solve_stage(
    newton_counter: qd.types.ndarray(qd.i32, ndim=0),
    pcg_counter: qd.types.ndarray(qd.i32, ndim=0),
    linesearch_counter: qd.types.ndarray(qd.i32, ndim=0),
    fem_state: FEMState,
    fem_info: FEMInfo,
    vertex_constraints: FEMVertexConstraints,
    fem_config: qd.template(),
    stage: qd.template(),
):
    """
    Run a stage of the implicit solve of a substep, or of its adjoint solve (see 'FEMSolver._solve_on_host').

    The adjoint solve seeds the adjoint of the forces with its solution (see '_implicit_solve_grad').
    """
    if qd.static(stage == FEMSolveStage.INIT_NEWTON):
        func_build_coarse(fem_state, fem_info, vertex_constraints, fem_config)
        fem_state.solve_loops.n_newton_left[()] = fem_info.n_newton_iterations[None]
    if qd.static(stage == FEMSolveStage.END_NEWTON):
        func_end_newton(newton_counter, fem_state, fem_info, fem_config)
    if qd.static(stage == FEMSolveStage.NEWTON_HEAD):
        func_newton_head(pcg_counter, fem_state, fem_info, vertex_constraints, fem_config)
    if qd.static(stage == FEMSolveStage.INIT_ADJOINT):
        func_activate_envs(fem_state)
        func_build_coarse(fem_state, fem_info, vertex_constraints, fem_config)
        func_init_pcg(pcg_counter, fem_state, fem_info, vertex_constraints, fem_config, is_adjoint=True)
    if qd.static(stage == FEMSolveStage.PCG_ITER):
        if qd.static(fem_config.is_pcg_solved_per_env_block):
            func_pcg_solve_per_env_block(pcg_counter, fem_state, fem_info, vertex_constraints, fem_config)
        else:
            func_pcg_iter(pcg_counter, fem_state, fem_info, vertex_constraints, fem_config)
    if qd.static(stage == FEMSolveStage.SEED_ADJOINT):
        func_seed_force_adjoint(fem_state)
    if qd.static(stage == FEMSolveStage.INIT_LINESEARCH):
        func_init_linesearch(linesearch_counter, fem_state, fem_info, fem_config)
    if qd.static(stage == FEMSolveStage.LINESEARCH_ITER):
        func_linesearch_iter(linesearch_counter, fem_state, fem_info, fem_config)
    if qd.static(stage == FEMSolveStage.SKIP_LINESEARCH):
        func_skip_linesearch(fem_state, fem_info)


@qd.kernel(fastcache=True)
def kernel_compute_implicit_force(
    fem_state: FEMState,
    fem_info: FEMInfo,
    vertex_constraints: FEMVertexConstraints,
    fem_config: qd.template(),
):
    """
    Compute the forces at the end of a substep of the implicit solver, whose autodiff alone runs (see
    '_implicit_solve_grad').

    The elements scatter their forces, which autodiff reverses, where the solve gathers. A constrained vertex has no
    force, while its motion strains the elements around it.
    """
    for i_e, i_b in qd.ndrange(
        *fem_state.elements_energy.energy.shape, axes=qd.static(eoa(fem_state.elements_energy.energy))
    ):
        _, gradient = func_element_energy_gradient_damped(i_e, i_b, fem_state, fem_info, fem_config)
        i_vs = fem_info.elements.el2v[i_e]
        S = func_element_mapping_matrix(i_e, fem_info)
        force = -fem_info.elements.V[i_e] * gradient @ S.transpose()
        has_constraints = vertex_constraints.envs_n_constraints[i_b] > 0
        for i_k in qd.static(range(4)):
            i_v = i_vs[i_k]
            if func_is_vertex_free(i_v, i_b, has_constraints, vertex_constraints):
                qd.atomic_add(fem_state.vertices_energy.force[i_v, i_b], force[:, i_k])
    for i_v, i_b in qd.ndrange(
        *fem_state.vertices_energy.force.shape, axes=qd.static(eoa(fem_state.vertices_energy.force))
    ):
        has_constraints = vertex_constraints.envs_n_constraints[i_b] > 0
        qd.atomic_add(
            fem_state.vertices_energy.force[i_v, i_b],
            func_vertex_inertia_force(i_v, i_b, has_constraints, fem_state, fem_info, vertex_constraints),
        )


@qd.kernel(fastcache=True)
def kernel_material_data_grad(
    fem_state: FEMState,
    fem_info: FEMInfo,
    vertex_constraints: FEMVertexConstraints,
    fem_config: qd.template(),
):
    """
    Carry the adjoint of the data the materials compute from the start of the substep to the start positions, element
    by element.

    The forces seeded with lambda pull on that data (see '_implicit_solve_grad').
    """
    for i_e, i_b in qd.ndrange(*fem_state.elements.actu.shape, axes=qd.static(eoa(fem_state.elements.actu))):
        i_vs = fem_info.elements.el2v[i_e]
        S = func_element_mapping_matrix(i_e, fem_info)
        # The forces -V * gradient @ S^T on the free vertices dotted with lambda give <gradient, -V * lambda @ S>
        vertices_lambda = qd.Matrix.zero(gs.qd_float, 3, 4)
        has_constraints = vertex_constraints.envs_n_constraints[i_b] > 0
        for i_k in qd.static(range(4)):
            i_v = i_vs[i_k]
            if func_is_vertex_free(i_v, i_b, has_constraints, vertex_constraints):
                vertices_lambda[:, i_k] = fem_state.vertices_energy.force.grad[i_v, i_b]
        gradient_grad = -fem_info.elements.V[i_e] * vertices_lambda @ S
        _, F_start = func_element_J_F(i_e, i_b, fem_state.vertices.pos, fem_info)
        _, F = func_element_J_F(i_e, i_b, fem_state.vertices.pos_next, fem_info)
        dF_damping = qd.Matrix.zero(gs.qd_float, 3, 3)
        if qd.static(fem_config.has_damping_beta):
            dF_damping = (
                fem_info.damping_beta_over_dt[None] * func_element_displacement(i_e, i_b, fem_state, fem_info) @ S
            )
        actu = fem_state.elements.actu[i_e, i_b]
        F_start_grad = func_element_precompute_grad(
            i_e, i_b, F_start, F, dF_damping, gradient_grad, actu, fem_state, fem_info, fem_config
        )
        # F_start = D B, D holding the positions of the first three vertices relative to the last in its columns
        D_grad = F_start_grad @ fem_info.elements.B[i_e].transpose()
        for i_k in qd.static(range(3)):
            qd.atomic_add(fem_state.vertices.pos.grad[i_vs[i_k], i_b], D_grad[:, i_k])
        qd.atomic_add(fem_state.vertices.pos.grad[i_vs[3], i_b], -D_grad[:, 0] - D_grad[:, 1] - D_grad[:, 2])


@qd.kernel(fastcache=True)
def kernel_setup_pos_vel(fem_state: FEMState, fem_info: FEMInfo):
    """Set the velocities of the vertices at the end of a substep of the implicit solver from their displacement."""
    for i_v, i_b in qd.ndrange(*fem_state.vertices.pos.shape, axes=qd.static(eoa(fem_state.vertices.vel))):
        fem_state.vertices.vel_next[i_v, i_b] = (
            fem_state.vertices.pos_next[i_v, i_b] - fem_state.vertices.pos[i_v, i_b]
        ) / fem_info.substep_dt[None]


@qd.kernel(fastcache=True)
def kernel_copy_next_to_curr(fem_state: FEMState):
    """Copy the positions and velocities the substep integrated to the state of the vertices."""
    for i_v, i_b in qd.ndrange(*fem_state.vertices.pos.shape, axes=qd.static(eoa(fem_state.vertices.pos))):
        fem_state.vertices.pos[i_v, i_b] = fem_state.vertices.pos_next[i_v, i_b]
        fem_state.vertices.vel[i_v, i_b] = fem_state.vertices.vel_next[i_v, i_b]


@qd.kernel(fastcache=True)
def kernel_get_forces(
    forces: qd.types.ndarray(),
    fem_state: FEMState,
    fem_info: FEMInfo,
    vertex_constraints: FEMVertexConstraints,
    fem_config: qd.template(),
):
    """Gather the internal elastic force on every vertex, one row per environment."""
    func_accumulate_vertices(
        forces,
        envs_is_active=None,
        pcg_counter=None,
        fem_state=fem_state,
        fem_info=fem_info,
        vertex_constraints=vertex_constraints,
        fem_config=fem_config,
        gather_Ap=False,
        gather_internal_force=True,
        update_loop_flag=False,
    )


# ------------------------------------------------------------------------------------
# --------------------------------------- io -----------------------------------------
# ------------------------------------------------------------------------------------


@qd.func
def func_add_vertices_and_surfaces(
    v_start: int,
    el_start: int,
    s_start: int,
    mass: float,
    friction_mu: float,
    verts: qd.types.ndarray(),
    tri2v: qd.types.ndarray(),
    tri2el: qd.types.ndarray(),
    fem_state: FEMState,
    fem_info: FEMInfo,
):
    """
    Add the vertices of an entity at rest and its surface triangles.

    The vertices start with mass `mass` and friction coefficient `friction_mu`, and the triangles reference the elements
    `tri2el` offset by `el_start`.
    """
    for i_v_, i_b in qd.ndrange(
        verts.shape[0], fem_state.vertices.pos.shape[1], axes=qd.static(eoa(fem_state.vertices.pos))
    ):
        i_v = i_v_ + v_start
        fem_state.vertices.pos[i_v, i_b] = qd.Vector([verts[i_v_, 0], verts[i_v_, 1], verts[i_v_, 2]])
        fem_state.vertices.vel[i_v, i_b] = qd.Vector.zero(gs.qd_float, 3)

    for i_v_ in range(verts.shape[0]):
        i_v = i_v_ + v_start
        fem_info.vertices.mass[i_v] = mass
        fem_info.vertices.mass_over_dt2[i_v] = 0.0
        fem_info.vertices.friction_mu[i_v] = friction_mu

    for i_s_ in range(tri2v.shape[0]):
        i_s = i_s_ + s_start
        fem_info.surfaces.tri2v[i_s] = qd.Vector(
            [tri2v[i_s_, i_k] + v_start for i_k in qd.static(range(3))], dt=gs.qd_int
        )
        fem_info.surfaces.tri2el[i_s] = tri2el[i_s_] + el_start


@qd.kernel(fastcache=True)
def kernel_add_elements(
    v_start: int,
    el_start: int,
    s_start: int,
    model: int,
    mu: float,
    lam: float,
    rho: float,
    friction_mu: float,
    muscle_stiffness: float,
    verts: qd.types.ndarray(),
    elems: qd.types.ndarray(),
    tri2v: qd.types.ndarray(),
    tri2el: qd.types.ndarray(),
    fem_state: FEMState,
    fem_info: FEMInfo,
    is_muscle: bool,
):
    """Add the vertices, elements and surface triangles of an entity at rest, with the parameters of its material."""
    func_add_vertices_and_surfaces(
        v_start, el_start, s_start, 0.0, friction_mu, verts, tri2v, tri2el, fem_state, fem_info
    )

    for i_e_ in range(elems.shape[0]):
        i_e = i_e_ + el_start
        i_vs = qd.Vector([elems[i_e_, i_k] + v_start for i_k in qd.static(range(4))], dt=gs.qd_int)
        pos_d = fem_state.vertices.pos[i_vs[3], 0]
        B_inv = qd.Matrix.cols([fem_state.vertices.pos[i_vs[i_k], 0] - pos_d for i_k in qd.static(range(3))])
        fem_info.elements.B[i_e] = B_inv.inverse()
        det = B_inv.determinant()
        # The determinant is negative for vertices in the expected order
        if det >= 0.0:
            fem_info.has_inverted_elements[()] = True
        V = qd.abs(det) / 6.0
        fem_info.elements.V[i_e] = V
        fem_info.elements.el2v[i_e] = i_vs
        fem_info.elements.model[i_e] = model
        fem_info.elements.mu[i_e] = mu
        fem_info.elements.lam[i_e] = lam
        fem_info.elements.friction_mu[i_e] = friction_mu
        fem_info.elements.mass_scaled[i_e] = rho * V * fem_info.vol_scale[None]
        fem_info.elements.is_muscle[i_e] = is_muscle
        fem_info.elements.muscle_stiffness[i_e] = muscle_stiffness
        fem_info.elements.muscle_group[i_e] = 0
        fem_info.elements.muscle_direction[i_e] = qd.Vector([0.0, 0.0, 1.0], dt=gs.qd_float)

    # The elements add their mass to their vertices in order, so the vertex masses are reproducible bitwise
    qd.loop_config(serialize=True)
    for i_e_ in range(elems.shape[0]):
        i_e = i_e_ + el_start
        mass = 0.25 * rho * fem_info.elements.V[i_e]
        for i_k in qd.static(range(4)):
            i_v = fem_info.elements.el2v[i_e][i_k]
            fem_info.vertices.mass[i_v] = fem_info.vertices.mass[i_v] + mass
            fem_info.vertices.mass_over_dt2[i_v] = fem_info.vertices.mass_over_dt2[i_v] + mass * fem_info.dt2_inv[None]

    for i_v_ in range(verts.shape[0]):
        i_v = i_v_ + v_start
        fem_info.vertices.mass_inv[i_v] = 1.0 / fem_info.vertices.mass[i_v]

    for i_e_, i_b in qd.ndrange(
        elems.shape[0], fem_state.vertices.pos.shape[1], axes=qd.static(eoa(fem_state.elements.actu))
    ):
        fem_state.elements.actu[i_e_ + el_start, i_b] = 0.0


@qd.kernel(fastcache=True)
def kernel_add_cloth(
    v_start: int,
    s_start: int,
    verts: qd.types.ndarray(),
    tri2v: qd.types.ndarray(),
    tri2el: qd.types.ndarray(),
    fem_state: FEMState,
    fem_info: FEMInfo,
):
    """Add the vertices and surface triangles of a cloth entity, each triangle being its own element."""
    func_add_vertices_and_surfaces(v_start, s_start, s_start, 1.0, 0.0, verts, tri2v, tri2el, fem_state, fem_info)


@qd.kernel(fastcache=True)
def kernel_add_vverts(
    vvert_start: int,
    vface_start: int,
    v_start: int,
    verts_idx: qd.types.ndarray(),
    uvs: qd.types.ndarray(element_dim=1),
    vfaces: qd.types.ndarray(element_dim=1),
    fem_info: FEMInfo,
):
    for i_vv_ in range(verts_idx.shape[0]):
        fem_info.vverts_vert_idx[i_vv_ + vvert_start] = verts_idx[i_vv_] + v_start
    for i_vv_ in range(uvs.shape[0]):
        fem_info.vverts_uvs[i_vv_ + vvert_start] = uvs[i_vv_]
    for i_vf_ in range(vfaces.shape[0]):
        fem_info.vfaces_indices[i_vf_ + vface_start] = vfaces[i_vf_] + vvert_start


@qd.kernel(fastcache=True)
def kernel_update_vverts_render(fem_state: FEMState, fem_info: FEMInfo):
    """Move the visual vertices to the position of the simulated vertex each one stands for, in single precision."""
    for i_vv, i_b in qd.ndrange(*fem_state.vverts_render.pos.shape, axes=qd.static(eoa(fem_state.vverts_render.pos))):
        i_v = fem_info.vverts_vert_idx[i_vv]
        fem_state.vverts_render.pos[i_vv, i_b] = qd.cast(fem_state.vertices.pos[i_v, i_b], qd.f32)


# ------------------------------------------------------------------------------------
# -------------------------------- vertex constraints --------------------------------
# ------------------------------------------------------------------------------------


@qd.kernel(fastcache=True)
def kernel_clear_vertex_constraints(vertex_constraints: FEMVertexConstraints):
    for i_v, i_b in qd.ndrange(
        *vertex_constraints.vertices_slot.shape, axes=qd.static(eoa(vertex_constraints.vertices_slot))
    ):
        vertex_constraints.vertices_slot[i_v, i_b] = -1
    for i_b in range(vertex_constraints.envs_n_constraints.shape[0]):
        vertex_constraints.envs_n_constraints[i_b] = 0


@qd.kernel(fastcache=True)
def kernel_set_vertex_constraints(
    verts_idx: qd.types.ndarray(),
    envs_idx: qd.types.ndarray(),
    link_idx: int,
    stiffness: float,
    target_poss: qd.types.ndarray(),
    link_init_pos: qd.types.ndarray(),
    link_init_quat: qd.types.ndarray(),
    fem_state: FEMState,
    vertex_constraints: FEMVertexConstraints,
    is_soft_constraint: bool,
):
    """
    Constrain some vertices, those following a link keeping their offset to where it stands.

    A vertex already constrained keeps its slot, and a new one takes the next free slot of its environment, the vertices
    of an environment taking their slots in order. Every selected environment has the slots left (see
    'FEMSolver.set_vertex_constraints').
    """
    for i_b_ in range(envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        for i_v_ in range(verts_idx.shape[1]):
            i_v = verts_idx[i_b_, i_v_]
            # The earlier occurrences of a repeated vertex are -1 (see 'drop_repeated_index')
            if i_v >= 0:
                i_s = vertex_constraints.vertices_slot[i_v, i_b]
                if i_s < 0:
                    i_s = vertex_constraints.envs_n_constraints[i_b]
                    vertex_constraints.envs_n_constraints[i_b] = i_s + 1
                    vertex_constraints.vertices_slot[i_v, i_b] = i_s
                    vertex_constraints.vertex_idx[i_s, i_b] = i_v
                vertex_constraints.is_soft_constraint[i_s, i_b] = is_soft_constraint
                vertex_constraints.stiffness[i_s, i_b] = stiffness
                vertex_constraints.link_idx[i_s, i_b] = link_idx
                vertex_constraints.target_pos[i_s, i_b] = qd.Vector(
                    [target_poss[i_b_, i_v_, 0], target_poss[i_b_, i_v_, 1], target_poss[i_b_, i_v_, 2]]
                )
                vertex_constraints.link_offset_pos[i_s, i_b] = fem_state.vertices.pos[i_v, i_b] - qd.Vector(
                    [link_init_pos[i_b_, 0], link_init_pos[i_b_, 1], link_init_pos[i_b_, 2]]
                )
                vertex_constraints.link_init_quat[i_s, i_b] = qd.Vector(
                    [
                        link_init_quat[i_b_, 0],
                        link_init_quat[i_b_, 1],
                        link_init_quat[i_b_, 2],
                        link_init_quat[i_b_, 3],
                    ]
                )


@qd.kernel(fastcache=True)
def kernel_update_constraint_targets(
    verts_idx: qd.types.ndarray(),
    envs_idx: qd.types.ndarray(),
    target_poss: qd.types.ndarray(),
    vertex_constraints: FEMVertexConstraints,
):
    """Set the targets of some constrained vertices, the free ones being left as they are."""
    for i_v_, i_b_ in qd.ndrange(verts_idx.shape[1], envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        i_v = verts_idx[i_b_, i_v_]
        if i_v >= 0:
            i_s = vertex_constraints.vertices_slot[i_v, i_b]
            if i_s >= 0:
                vertex_constraints.target_pos[i_s, i_b] = qd.Vector(
                    [target_poss[i_b_, i_v_, 0], target_poss[i_b_, i_v_, 1], target_poss[i_b_, i_v_, 2]]
                )


@qd.kernel(fastcache=True)
def kernel_remove_vertex_constraints(
    verts_idx: qd.types.ndarray(), envs_idx: qd.types.ndarray(), vertex_constraints: FEMVertexConstraints
):
    """Free some vertices, the last constraint of their environment moving into the slot of each one."""
    for i_b_ in range(envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        for i_v_ in range(verts_idx.shape[1]):
            i_v = verts_idx[i_b_, i_v_]
            i_s = -1
            if i_v >= 0:
                i_s = vertex_constraints.vertices_slot[i_v, i_b]
            if i_s >= 0:
                i_s_last = vertex_constraints.envs_n_constraints[i_b] - 1
                i_v_last = vertex_constraints.vertex_idx[i_s_last, i_b]
                vertex_constraints.vertex_idx[i_s, i_b] = i_v_last
                vertex_constraints.target_pos[i_s, i_b] = vertex_constraints.target_pos[i_s_last, i_b]
                vertex_constraints.is_soft_constraint[i_s, i_b] = vertex_constraints.is_soft_constraint[i_s_last, i_b]
                vertex_constraints.stiffness[i_s, i_b] = vertex_constraints.stiffness[i_s_last, i_b]
                vertex_constraints.link_idx[i_s, i_b] = vertex_constraints.link_idx[i_s_last, i_b]
                vertex_constraints.link_offset_pos[i_s, i_b] = vertex_constraints.link_offset_pos[i_s_last, i_b]
                vertex_constraints.link_init_quat[i_s, i_b] = vertex_constraints.link_init_quat[i_s_last, i_b]
                vertex_constraints.vertices_slot[i_v_last, i_b] = i_s
                vertex_constraints.vertices_slot[i_v, i_b] = -1
                vertex_constraints.envs_n_constraints[i_b] = i_s_last
