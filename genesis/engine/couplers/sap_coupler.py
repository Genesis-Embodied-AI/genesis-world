import dataclasses
import math
import textwrap
from typing import TYPE_CHECKING

import numpy as np

import igl

import quadrants as qd

import genesis as gs
import genesis.utils.element as eu
import genesis.utils.geom as gu
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
from genesis.engine.solvers.fem_solver import (
    ELEMENTS,
    VERTICES,
    FEMInfo,
    FEMState,
    FEMVertexConstraints,
    func_elastic_product,
    func_element_mapping_matrix,
)
from genesis.engine.solvers.rigid.abd.forward_kinematics import kernel_update_all_verts
from genesis.engine.solvers.rigid.data import (
    DOFS,
    GEOMS,
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
from genesis.utils.data import ENVS, Config, Data, DataKind, Size, eoa, field
from genesis.utils.misc import qd_to_numpy, tensor_to_array

if TYPE_CHECKING:
    from genesis.engine.simulator import Simulator


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

# Cosine threshold for whether two vectors are considered to be in the same direction. Set to zero for strictly positive.
COS_ANGLE_THRESHOLD = math.cos(math.pi * 5.0 / 8.0)

# An estimate of the maximum number of contact pairs per AABB query.
MAX_N_QUERY_RESULT_PER_AABB = 32

# The vertices and elements of the tetrahedral volume meshes of the rigid geoms, and the rigid triangles touching FEM
RIGID_VOLUME_VERTS = Size("rigid_volume_verts")
RIGID_VOLUME_ELEMS = Size("rigid_volume_elems")
RIGID_TRI_FACES = Size("rigid_tri_faces")

# Labels of the arrays that the solvers and the contact types of a scene leave unused
FEM = "fem"
RIGID = "rigid"
FEM_PRESSURE = "fem_pressure"
RIGID_VOLUME = "rigid_volume"
RIGID_RIGID = "rigid_rigid"
RIGID_FEM = "rigid_fem"
TET_TABLES = "tet_tables"


class SAPFEMVertices(Data, kind=DataKind.SCRATCH, shape=(VERTICES, ENVS)):
    """The velocities of the FEM vertices that the SAP solve iterates on, with the vectors of its solves.

    The vectors serve its preconditioned conjugate gradient (PCG) solve and its line search.
    """

    v: qd.Tensor = field((float, 3))
    # Difference between the velocity and the free-motion one
    v_diff: qd.Tensor = field((float, 3))
    gradient: qd.Tensor = field((float, 3))
    impulse: qd.Tensor = field((float, 3))
    # Diagonal 3x3 block of the Hessian, and its inverse as preconditioner
    diag3x3: qd.Tensor = field((float, (3, 3)))
    prec: qd.Tensor = field((float, (3, 3)))
    x: qd.Tensor = field((float, 3))
    r: qd.Tensor = field((float, 3))
    z: qd.Tensor = field((float, 3))
    p: qd.Tensor = field((float, 3))
    Ap: qd.Tensor = field((float, 3))
    # Velocity at the start of the line search, and the product of the Hessian with the search direction
    x_prev: qd.Tensor = field((float, 3))
    dp: qd.Tensor = field((float, 3))


class SAPRigidDofs(Data, kind=DataKind.SCRATCH, shape=(DOFS, ENVS)):
    """The velocities of the rigid dofs that the SAP solve iterates on, with the vectors of its solves.

    The vectors serve its preconditioned conjugate gradient (PCG) solve and its line search.
    """

    v: qd.Tensor = field(float)
    # Difference between the velocity and the free-motion one, and its product with the mass matrix
    v_diff: qd.Tensor = field(float)
    mass_v_diff: qd.Tensor = field(float)
    gradient: qd.Tensor = field(float)
    impulse: qd.Tensor = field(float)
    x: qd.Tensor = field(float)
    r: qd.Tensor = field(float)
    z: qd.Tensor = field(float)
    p: qd.Tensor = field(float)
    Ap: qd.Tensor = field(float)
    # Velocity at the start of the line search, and the product of the Hessian with the search direction
    x_prev: qd.Tensor = field(float)
    dp: qd.Tensor = field(float)


class SAPEnvs(Data, kind=DataKind.SCRATCH, shape=(ENVS,)):
    """The convergence state of the SAP solve of each environment, with the scalars of its solves.

    The scalars serve its preconditioned conjugate gradient (PCG) solve and its line search.
    """

    is_active: qd.Tensor = field(bool)
    is_pcg_active: qd.Tensor = field(bool)
    is_linesearch_active: qd.Tensor = field(bool)
    gradient_norm: qd.Tensor = field(float)
    momentum_norm: qd.Tensor = field(float)
    impulse_norm: qd.Tensor = field(float)
    rTr: qd.Tensor = field(float)
    rTz: qd.Tensor = field(float)
    rTr_new: qd.Tensor = field(float)
    rTz_new: qd.Tensor = field(float)
    pTAp: qd.Tensor = field(float)
    alpha: qd.Tensor = field(float)
    beta: qd.Tensor = field(float)
    prev_energy: qd.Tensor = field(float)
    energy: qd.Tensor = field(float)
    step_size: qd.Tensor = field(float)
    m: qd.Tensor = field(float)
    # First and second derivatives of the total energy along the step size, the latter of its dynamic part alone too
    dell_dalpha: qd.Tensor = field(float)
    d2ellA_dalpha2: qd.Tensor = field(float)
    d2ell_dalpha2: qd.Tensor = field(float)
    dell_scale: qd.Tensor = field(float)
    # The bracket of the step size, its tolerance, and the scaled derivative of the energy at its bounds and at the step
    alpha_min: qd.Tensor = field(float)
    alpha_max: qd.Tensor = field(float)
    alpha_tol: qd.Tensor = field(float)
    f_lower: qd.Tensor = field(float)
    f_upper: qd.Tensor = field(float)
    f: qd.Tensor = field(float)
    df: qd.Tensor = field(float)
    minus_dalpha: qd.Tensor = field(float)
    minus_dalpha_prev: qd.Tensor = field(float)


class SAPState(Data, kind=DataKind.SCRATCH):
    """The data of the SAP coupler that a substep writes: the iterates of its solve and the posed pressure fields."""

    fem_vertices: SAPFEMVertices = field(labels=FEM)
    rigid_dofs: SAPRigidDofs = field(labels=RIGID)
    envs: SAPEnvs
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
    """The model of the SAP coupler: hydroelastic pressure fields, rigid volume meshes and solve constants."""

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
    n_dofs: qd.Tensor = field(int, shape=())
    # The rigid ground plane, -1 without one (see 'SAPCoupler.build')
    ground_geom_idx: qd.Tensor = field(int, shape=())
    substep_dt: qd.Tensor = field(float, shape=())
    dt2: qd.Tensor = field(float, shape=())
    convergence_atol: qd.Tensor = field(float, shape=())
    convergence_rtol: qd.Tensor = field(float, shape=())
    taud: qd.Tensor = field(float, shape=())
    sigma: qd.Tensor = field(float, shape=())
    # The near-rigid factor of the contact regularization, beta^2 / (4 pi^2)
    beta_factor: qd.Tensor = field(float, shape=())
    pcg_threshold: qd.Tensor = field(float, shape=())
    linesearch_ftol: qd.Tensor = field(float, shape=())
    linesearch_max_step_size: qd.Tensor = field(float, shape=())
    hydroelastic_stiffness: qd.Tensor = field(float, shape=())
    point_contact_stiffness: qd.Tensor = field(float, shape=())
    rigid_floor_vert_stiffness: qd.Tensor = field(float, shape=())
    cos_angle_threshold: qd.Tensor = field(float, shape=())
    contact_eps: qd.Tensor = field(float, shape=())
    EPS: qd.Tensor = field(float, shape=())


class SAPConfig(Config):
    """The compile-time configuration of the kernels of the SAP coupler."""

    has_fem: bool
    has_rigid: bool
    has_equality_constraints: bool


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


@qd.func
def tri_barycentric(p, tri_vertices, normal):
    """
    Compute the barycentric coordinates of point p with respect to the triangle defined by tri_vertices.

    Parameters
    ----------
    p:
        The point in space for which to compute barycentric coordinates.
    tri_vertices:
        a matrix of shape (3, 3) where each column is a vertex of the triangle.
    normal:
        the normal vector of the triangle.

    Notes
    -----
    This function assumes that the triangle is not degenerated.
    """
    v0 = tri_vertices[:, 0]
    v1 = tri_vertices[:, 1]
    v2 = tri_vertices[:, 2]

    # Compute the areas of the triangles formed by the vertices
    area_tri_inv = 1.0 / (v1 - v0).cross((v2 - v0)).dot(normal)

    # Compute the barycentric coordinates
    b0 = (v2 - v1).cross(p - v1).dot(normal) * area_tri_inv
    b1 = (v0 - v2).cross(p - v2).dot(normal) * area_tri_inv
    b2 = 1.0 - b0 - b1

    return gs.qd_vec3(b0, b1, b2)


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


class SAPCoupler(RBC):
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
        self._n_sap_iterations = options.n_sap_iterations
        self._n_pcg_iterations = options.n_pcg_iterations
        self._n_linesearch_iterations = options.n_linesearch_iterations
        self._sap_convergence_atol = options.sap_convergence_atol
        self._sap_convergence_rtol = options.sap_convergence_rtol
        self._sap_taud = options.sap_taud
        self._sap_beta = options.sap_beta
        self._sap_sigma = options.sap_sigma
        self._pcg_threshold = options.pcg_threshold
        self._linesearch_ftol = options.linesearch_ftol
        self._linesearch_max_step_size = options.linesearch_max_step_size
        self._hydroelastic_stiffness = options.hydroelastic_stiffness
        self._point_contact_stiffness = options.point_contact_stiffness
        if gs.qd_float == qd.f32:
            gs.raise_exception(
                "SAPCoupler does not support 32bits precision. Please specify precision='64' when initializing Genesis."
            )
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
        self.sap_state: SAPState | None = None
        self.sap_info: SAPInfo | None = None
        self.sap_config: SAPConfig | None = None
        # The contact handlers in the order the kernels run them, and the handler of the equality constraints, if any
        self.contact_handlers: tuple["BaseContactHandler", ...] = ()
        self.equality_constraint_handler: "RigidConstraintHandler | None" = None

    # ------------------------------------------------------------------------------------
    # --------------------------------- Initialization -----------------------------------
    # ------------------------------------------------------------------------------------

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

        if self.rigid_solver.is_active:
            # A ground plane alone leaves no rigid volume to put in contact
            rigid_collision_geoms_idx = [
                geom.idx for geom in self.rigid_solver.geoms if geom.contype or geom.conaffinity
            ]
            if rigid_collision_geoms_idx == [ground_geom_idx]:
                self._rigid_floor_contact_type = RigidFloorContactType.NONE
                self._rigid_rigid_contact_type = RigidRigidContactType.NONE
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

        if self._has_fem_pressure:
            fem_pressure = []
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

        n_rigid_volume_verts = n_rigid_volume_elems = 0
        if self._rigid_compliant:
            rigid_volume_verts = []
            rigid_volume_elems = []
            rigid_volume_verts_geom_idx = []
            rigid_volume_elems_geom_idx = []
            rigid_pressure_field = []
            for geom in self.rigid_solver.geoms:
                # The rigid volumes leave the ground plane out (see 'ground_geom_idx' above)
                if (geom.contype or geom.conaffinity) and geom.idx != ground_geom_idx:
                    volume = geom.trimesh.volume
                    tet_cfg = {"nobisect": False, "maxvolume": volume / 100}
                    mesh_verts, mesh_elems = eu.mesh_to_elements(geom.trimesh, tet_cfg=tet_cfg)
                    verts, elems = eu.split_all_surface_tets(mesh_verts, mesh_elems)
                    rigid_volume_verts.append(verts)
                    rigid_volume_elems.append(elems + n_rigid_volume_verts)
                    rigid_volume_verts_geom_idx.append(np.full(len(verts), geom.idx, dtype=gs.np_int))
                    rigid_volume_elems_geom_idx.append(np.full(len(elems), geom.idx, dtype=gs.np_int))
                    rigid_pressure_field.append(
                        compute_pressure_field(verts, geom.init_verts, geom.init_faces, self._hydroelastic_stiffness)
                    )
                    n_rigid_volume_verts += len(verts)
                    n_rigid_volume_elems += len(elems)
            if not rigid_volume_verts:
                gs.raise_exception("No rigid collision geometries found.")
        if has_rigid_rigid_contact:
            rigid_collision_pair_idx = qd_to_numpy(self.rigid_solver.collider.collider_info.collision_pair_idx)

        sizes = dict(
            vertices=self.fem_solver.n_vertices,
            rigid_volume_verts=n_rigid_volume_verts,
            rigid_volume_elems=n_rigid_volume_elems,
        )
        self.sap_state = SAPState(
            sizes=dict(sizes, envs=self._B, elements=self.fem_solver.n_elements, dofs=self.rigid_solver.n_dofs),
            unused=dict(
                fem=not self.fem_solver.is_active,
                rigid=not self.rigid_solver.is_active,
                fem_pressure=not self._has_fem_pressure,
                rigid_volume=not self._rigid_compliant,
            ),
        )
        self.sap_info = SAPInfo(
            sizes=dict(sizes, rigid_tri_faces=len(rigid_tri_faces_idx), geoms=self.rigid_solver.n_geoms),
            values=dict(
                n_dofs=self.rigid_solver.n_dofs,
                ground_geom_idx=ground_geom_idx,
                substep_dt=self.sim._substep_dt,
                dt2=self.sim._substep_dt**2,
                convergence_atol=self._sap_convergence_atol,
                convergence_rtol=self._sap_convergence_rtol,
                taud=self._sap_taud,
                sigma=self._sap_sigma,
                beta_factor=self._sap_beta**2 / (4.0 * qd.math.pi**2),
                pcg_threshold=self._pcg_threshold,
                linesearch_ftol=self._linesearch_ftol,
                linesearch_max_step_size=self._linesearch_max_step_size,
                hydroelastic_stiffness=self._hydroelastic_stiffness,
                point_contact_stiffness=self._point_contact_stiffness,
                rigid_floor_vert_stiffness=1.0e6,
                cos_angle_threshold=COS_ANGLE_THRESHOLD,
                contact_eps=1e-10,
                EPS=gs.EPS,
            ),
            unused=dict(
                fem_pressure=not self._has_fem_pressure,
                rigid_volume=not self._rigid_compliant,
                rigid_rigid=not has_rigid_rigid_contact,
                rigid_fem=not self._enable_rigid_fem_contact,
                tet_tables=not has_tet_tables,
            ),
        )
        if self._has_fem_pressure:
            self.sap_info.fem_pressure.from_numpy(np.concatenate(fem_pressure))
        if self._rigid_compliant:
            self.sap_info.rigid_volume_verts_rest.from_numpy(np.concatenate(rigid_volume_verts, dtype=gs.np_float))
            self.sap_info.rigid_volume_verts_geom_idx.from_numpy(np.concatenate(rigid_volume_verts_geom_idx))
            self.sap_info.rigid_pressure_field.from_numpy(np.concatenate(rigid_pressure_field, dtype=gs.np_float))
            self.sap_info.rigid_volume_elems.from_numpy(np.concatenate(rigid_volume_elems, dtype=gs.np_int))
            self.sap_info.rigid_volume_elems_geom_idx.from_numpy(np.concatenate(rigid_volume_elems_geom_idx))
            kernel_compute_rigid_pressure_gradient_rest(self.sap_info)
        if has_rigid_rigid_contact:
            self.sap_info.rigid_collision_pair_idx.from_numpy(rigid_collision_pair_idx)
        if self._enable_rigid_fem_contact:
            self.sap_info.rigid_tri_faces_idx.from_numpy(np.array(rigid_tri_faces_idx, dtype=gs.np_int))
        if has_tet_tables:
            self.sap_info.marching_tets_edge_table.from_numpy(np.array(MARCHING_TETS_EDGE_TABLE, dtype=gs.np_int))
            self.sap_info.tet_edges.from_numpy(np.array(TET_EDGES, dtype=gs.np_int))
        self.sap_config = SAPConfig(
            has_fem=self.fem_solver.is_active,
            has_rigid=self.rigid_solver.is_active,
            has_equality_constraints=self._has_equality_constraints,
        )

        # The contact handlers accumulate into the same gradients, so their order sets the rounding of the solve
        contact_handlers = []
        if self.fem_solver.is_active:
            if self._fem_floor_contact_type == FEMFloorContactType.TET:
                contact_handlers.append(FEMFloorTetContactHandler(self.sim))
            if self._fem_floor_contact_type == FEMFloorContactType.VERT:
                contact_handlers.append(FEMFloorVertContactHandler(self.sim))
            if self._enable_fem_self_tet_contact:
                contact_handlers.append(FEMSelfTetContactHandler(self.sim))
        if self.rigid_solver.is_active:
            if self._rigid_floor_contact_type == RigidFloorContactType.VERT:
                contact_handlers.append(RigidFloorVertContactHandler(self.sim))
            elif self._rigid_floor_contact_type == RigidFloorContactType.TET:
                contact_handlers.append(RigidFloorTetContactHandler(self.sim, n_rigid_volume_elems))
            if has_rigid_rigid_contact:
                contact_handlers.append(RigidRigidTetContactHandler(self.sim, n_rigid_volume_elems))
            if self._has_equality_constraints:
                self.equality_constraint_handler = RigidConstraintHandler(self.sim)
                kernel_build_equality_constraints(
                    self.rigid_solver.dyn_info.equalities,
                    self.rigid_solver.dyn_info.joints,
                    self.rigid_solver.rigid_info,
                    self.sap_info,
                    self.rigid_solver.rigid_config,
                    self.equality_constraint_handler,
                )
        if self._enable_rigid_fem_contact:
            contact_handlers.append(RigidFemTriTetContactHandler(self.sim))
        self.contact_handlers = tuple(contact_handlers)

        self._init_bvh(n_rigid_volume_elems if has_rigid_rigid_contact else 0)

    def reset(self, envs_idx=None):
        pass

    def _init_bvh(self, n_rigid_tets: int) -> None:
        """Allocate the tree set and the box query of each contact handler traversing a tree."""
        # The contact kernel takes the queries as one struct, so a disabled handler keeps its members, switched off
        # The surface tet boxes serve the self-contact tree and the rigid triangle query alike
        has_fem_tet_tree = self._enable_fem_self_tet_contact or self._enable_rigid_fem_contact
        n_tets = self.fem_solver.n_surface_elements if has_fem_tet_tree else 0
        self.fem_surface_tet_bvh_state, self.fem_surface_tet_bvh_config = get_bvh_data(
            self.sim._B, n_tets, is_active=has_fem_tet_tree
        )
        max_results = min(n_tets * MAX_N_QUERY_RESULT_PER_AABB * self.sim._B, 0x7FFFFFFF)
        # The surface tets of each environment queried against their own tree
        fem_self_query = BVHQueryState(
            leaves=self.fem_surface_tet_bvh_state.leaves,
            tree=self.fem_surface_tet_bvh_state.tree,
            results=BVHQueryResults(
                sizes=dict(query_results=max_results), unused=not self._enable_fem_self_tet_contact
            ),
        )

        n_rigid_tris = self.sap_info.rigid_tri_faces_idx.shape[0] if self._enable_rigid_fem_contact else 0
        self.rigid_tri_bvh_state, self.rigid_tri_bvh_config = get_bvh_data(
            self.sim._B, n_rigid_tris, is_active=self._enable_rigid_fem_contact
        )
        if self._enable_rigid_fem_contact:
            max_n_query_results_per_tri = max(n_rigid_tris, n_tets) * MAX_N_QUERY_RESULT_PER_AABB // n_rigid_tris
            max_results = min(n_rigid_tris * max_n_query_results_per_tri * self.sim._B, 0x7FFFFFFF)
        else:
            max_results = 0
        # The surface tets of each environment queried against the rigid triangle tree of that environment
        rigid_tri_query = BVHQueryState(
            leaves=self.fem_surface_tet_bvh_state.leaves,
            tree=self.rigid_tri_bvh_state.tree,
            results=BVHQueryResults(sizes=dict(query_results=max_results), unused=not self._enable_rigid_fem_contact),
        )

        has_rigid_tet_tree = n_rigid_tets > 0
        self.rigid_tet_bvh_state, self.rigid_tet_bvh_config = get_bvh_data(
            self.sim._B, n_rigid_tets, is_active=has_rigid_tet_tree
        )
        max_results = min(n_rigid_tets * MAX_N_QUERY_RESULT_PER_AABB * self.sim._B, 0x7FFFFFFF)
        # The rigid tets of each environment queried against their own tree
        rigid_tet_query = BVHQueryState(
            leaves=self.rigid_tet_bvh_state.leaves,
            tree=self.rigid_tet_bvh_state.tree,
            results=BVHQueryResults(sizes=dict(query_results=max_results), unused=not has_rigid_tet_tree),
        )
        self.contact_queries_state = SAPContactQueriesState(
            fem_self=fem_self_query, rigid_tri=rigid_tri_query, rigid_tet=rigid_tet_query
        )

    # ------------------------------------------------------------------------------------
    # -------------------------------------- Main ----------------------------------------
    # ------------------------------------------------------------------------------------

    def preprocess(self, i_step):
        rigid_solver, fem_solver = self.rigid_solver, self.fem_solver
        if self._has_fem_pressure:
            kernel_compute_fem_pressure_gradient(
                i_step, fem_solver.fem_state, self.sap_state, fem_solver.fem_info, self.sap_info
            )
        if rigid_solver.is_active:
            kernel_update_all_verts(rigid_solver.dyn_state, rigid_solver.dyn_info, rigid_solver.rigid_config)
        if self._rigid_compliant:
            kernel_update_rigid_volume_verts_pressure_gradient(
                rigid_solver.dyn_state.geoms, self.sap_state, self.sap_info
            )

        if self._enable_fem_self_tet_contact or self._enable_rigid_fem_contact:
            kernel_compute_fem_surface_tet_aabb(
                i_step, fem_solver.fem_state, self.fem_surface_tet_bvh_state.leaves, fem_solver.fem_info
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
        if rigid_solver.is_active and self._rigid_rigid_contact_type == RigidRigidContactType.TET:
            kernel_compute_rigid_tet_aabb(self.sap_state, self.rigid_tet_bvh_state.leaves, self.sap_info)
            build_bvh(self.rigid_tet_bvh_state, self.rigid_tet_bvh_config, eps=gs.EPS)

        self.has_contact, has_overflow = kernel_update_contact(
            i_step,
            fem_solver.fem_state,
            rigid_solver.dyn_state.dofs,
            rigid_solver.dyn_state.links,
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
            self.contact_handlers,
        )
        if has_overflow:
            message = "Overflowed In Contact Query: \n"
            for contact in self.contact_handlers:
                if contact.n_contact_pairs[None] > contact.max_contact_pairs:
                    message += (
                        f"{contact.name} max contact pairs: {contact.max_contact_pairs}"
                        f", using {contact.n_contact_pairs[None]}\n"
                    )
            gs.raise_exception(message)
        kernel_compute_regularization(
            fem_solver.fem_state,
            fem_solver.fem_info,
            rigid_solver.dyn_info.entities,
            rigid_solver.dyn_info.equalities,
            rigid_solver.rigid_info,
            self.sap_info,
            self.sap_config,
            self.contact_handlers,
            self.equality_constraint_handler,
        )

    def couple(self, i_step):
        # Equality constraints bind the rigid dofs whether anything is in contact or not
        if self.has_contact or self._has_equality_constraints:
            self.sap_solve(i_step)
            kernel_update_vel(
                i_step,
                self.fem_solver.fem_state,
                self.rigid_solver.dyn_state.dofs,
                self.sap_state,
                self.sap_info,
                self.sap_config,
            )

    # ------------------------------------------------------------------------------------
    # ------------------------------------- Solve ----------------------------------------
    # ------------------------------------------------------------------------------------

    def sap_solve(self, i_step):
        fem_state, fem_info = self.fem_solver.fem_state, self.fem_solver.fem_info
        vertex_constraints, fem_config = self.fem_solver.vertex_constraints, self.fem_solver.fem_config
        dofs_state = self.rigid_solver.dyn_state.dofs
        entities_info, rigid_info = self.rigid_solver.dyn_info.entities, self.rigid_solver.rigid_info
        sap_state, sap_info, sap_config = self.sap_state, self.sap_info, self.sap_config
        contact_handlers, constraint_handler = self.contact_handlers, self.equality_constraint_handler

        kernel_init_sap_solve(i_step, fem_state, dofs_state, sap_state, sap_info, sap_config)
        for i_iter in range(self._n_sap_iterations):
            if self.fem_solver.is_active:
                kernel_init_fem_unconstrained_gradient_diag(i_step, fem_state, sap_state, fem_info)
            if self.rigid_solver.is_active:
                kernel_init_rigid_unconstrained_gradient(dofs_state, sap_state, sap_info)
            # The first iteration starts from the free-motion velocity v*, so that A(v - v*) = 0
            if i_iter > 0:
                if self.fem_solver.is_active:
                    kernel_compute_fem_unconstrained_gradient(
                        fem_state, sap_state, fem_info, vertex_constraints, fem_config
                    )
                if self.rigid_solver.is_active:
                    kernel_compute_rigid_unconstrained_gradient(sap_state, rigid_info, sap_info)

            kernel_compute_constraint_contact_gradient_hessian_diag_prec(
                sap_state,
                fem_info,
                vertex_constraints,
                sap_info,
                fem_config,
                sap_config,
                contact_handlers,
                constraint_handler,
            )
            kernel_check_sap_convergence(sap_state, fem_info, rigid_info, sap_info, sap_config)

            kernel_init_pcg_solve(sap_state, entities_info, rigid_info, sap_info, sap_config)
            for _ in range(self._n_pcg_iterations):
                kernel_one_pcg_iter(
                    fem_state,
                    sap_state,
                    fem_info,
                    vertex_constraints,
                    entities_info,
                    rigid_info,
                    sap_info,
                    fem_config,
                    sap_config,
                    contact_handlers,
                    constraint_handler,
                )

            # Exact line search by a safeguarded Newton iteration (rtsafe) on the derivative of the cost along the
            # search direction.
            kernel_init_exact_linesearch(
                i_step,
                fem_state,
                dofs_state,
                sap_state,
                fem_info,
                vertex_constraints,
                rigid_info,
                sap_info,
                fem_config,
                sap_config,
                contact_handlers,
                constraint_handler,
            )
            for _ in range(self._n_linesearch_iterations):
                kernel_one_exact_linesearch_iter(
                    i_step,
                    fem_state,
                    dofs_state,
                    sap_state,
                    fem_info,
                    sap_info,
                    sap_config,
                    contact_handlers,
                    constraint_handler,
                )

    # ------------------------------------------------------------------------------------
    # ----------------------------------- Properties -------------------------------------
    # ------------------------------------------------------------------------------------
    @property
    def active_solvers(self):
        """All the active solvers managed by the scene's simulator."""
        return self.sim.active_solvers


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
            if qd.abs(signed_volume) > sap_info.EPS[None]:
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
        R = gu.qd_quat_to_R(quat, sap_info.EPS[None])
        sap_state.rigid_volume_verts[i_v, i_b] = R @ sap_info.rigid_volume_verts_rest[i_v] + pos

    for i_e, i_b in qd.ndrange(
        sap_info.rigid_volume_elems.shape[0],
        sap_state.rigid_pressure_gradient.shape[1],
        axes=qd.static(eoa(sap_state.rigid_pressure_gradient)),
    ):
        i_g = sap_info.rigid_volume_elems_geom_idx[i_e]
        pos = geoms_state.pos[i_g, i_b]
        quat = geoms_state.quat[i_g, i_b]
        R = gu.qd_quat_to_R(quat, sap_info.EPS[None])
        sap_state.rigid_pressure_gradient[i_e, i_b] = R @ sap_info.rigid_pressure_gradient_rest[i_e]


@qd.kernel
def kernel_compute_fem_pressure_gradient(
    i_step: int, fem_state: FEMState, sap_state: SAPState, fem_info: FEMInfo, sap_info: SAPInfo
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
            pos_v0 = fem_state.vertices.pos[i_step, i_v0, i_b]
            pos_v1 = fem_state.vertices.pos[i_step, i_v1, i_b]
            pos_v2 = fem_state.vertices.pos[i_step, i_v2, i_b]
            pos_v3 = fem_state.vertices.pos[i_step, i_v3, i_b]

            e10 = pos_v0 - pos_v1
            e12 = pos_v2 - pos_v1
            e13 = pos_v3 - pos_v1

            area_vector = e12.cross(e13)
            signed_volume = area_vector.dot(e10)
            if qd.abs(signed_volume) > sap_info.EPS[None]:
                grad_i = area_vector / signed_volume
                sap_state.fem_pressure_gradient[i_e, i_b] = (
                    sap_state.fem_pressure_gradient[i_e, i_b] + grad_i * sap_info.fem_pressure[i_v0]
                )


@qd.kernel
def kernel_compute_fem_surface_tet_aabb(i_step: int, fem_state: FEMState, bvh_leaves: BVHLeaves, fem_info: FEMInfo):
    """Fit the box of every FEM surface tet, the leaves of the trees of the FEM contacts."""
    for i_se, i_b in qd.ndrange(
        fem_info.surface_elements.shape[0], bvh_leaves.aabbs_min.shape[0], axes=qd.static(eoa(fem_state.vertices.pos))
    ):
        i_e = fem_info.surface_elements[i_se]
        i_vs = fem_info.elements.el2v[i_e]

        leaf_min = qd.Vector([np.inf, np.inf, np.inf], dt=gs.qd_float)
        leaf_max = -leaf_min
        for i in qd.static(range(4)):
            pos_v = fem_state.vertices.pos[i_step, i_vs[i], i_b]
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
    """Fit the box of every rigid volume element, the leaves of the trees of the rigid-rigid contacts."""
    for i_e, i_b in qd.ndrange(
        sap_info.rigid_volume_elems.shape[0],
        bvh_leaves.aabbs_min.shape[0],
        axes=qd.static(eoa(sap_state.rigid_volume_verts)),
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


@qd.kernel
def kernel_update_contact(
    i_step: int,
    fem_state: FEMState,
    dofs_state: DofsState,
    links_state: LinksState,
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
    contact_handlers: qd.template(),
) -> tuple[bool, bool]:
    """Detect the contacts of every handler and compute their Jacobians.

    Return whether any contact is found, and whether a handler found more of them than it can hold.
    """
    has_contact = False
    has_overflow = False
    for contact in qd.static(contact_handlers):
        has_overflow |= contact.detection(
            i_step,
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
        )
        has_contact |= contact.n_contact_pairs[None] > 0
        contact.compute_jacobian(dofs_state, links_state, links_info, sap_info, rigid_config)
    return has_contact, has_overflow


@qd.kernel
def kernel_compute_regularization(
    fem_state: FEMState,
    fem_info: FEMInfo,
    entities_info: EntitiesInfo,
    equalities_info: EqualitiesInfo,
    rigid_info: RigidInfo,
    sap_info: SAPInfo,
    sap_config: qd.template(),
    contact_handlers: qd.template(),
    constraint_handler: qd.template(),
):
    """Compute the regularization of every contact and equality constraint from its Delassus operator."""
    for contact in qd.static(contact_handlers):
        contact.compute_regularization(fem_state, fem_info, entities_info, rigid_info, sap_info)
    if qd.static(sap_config.has_equality_constraints):
        constraint_handler.compute_regularization(entities_info, equalities_info, rigid_info, sap_info)


@qd.kernel
def kernel_update_vel(
    i_step: int,
    fem_state: FEMState,
    dofs_state: DofsState,
    sap_state: SAPState,
    sap_info: SAPInfo,
    sap_config: qd.template(),
):
    """Write the velocities the SAP solve converged to back into the solvers."""
    if qd.static(sap_config.has_fem):
        for i_v, i_b in qd.ndrange(
            sap_state.fem_vertices.v.shape[0],
            sap_state.envs.is_active.shape[0],
            axes=qd.static(eoa(fem_state.vertices.vel)),
        ):
            fem_state.vertices.vel[i_step + 1, i_v, i_b] = sap_state.fem_vertices.v[i_v, i_b]
    if qd.static(sap_config.has_rigid):
        for i_d, i_b in qd.ndrange(
            sap_info.n_dofs[None], sap_state.envs.is_active.shape[0], axes=qd.static(eoa(dofs_state.vel))
        ):
            dofs_state.vel[i_d, i_b] = sap_state.rigid_dofs.v[i_d, i_b]


@qd.kernel
def kernel_init_sap_solve(
    i_step: int,
    fem_state: FEMState,
    dofs_state: DofsState,
    sap_state: SAPState,
    sap_info: SAPInfo,
    sap_config: qd.template(),
):
    """Start the SAP solve of every environment from the free-motion velocities of the solvers."""
    if qd.static(sap_config.has_fem):
        for i_v, i_b in qd.ndrange(
            sap_state.fem_vertices.v.shape[0],
            sap_state.envs.is_active.shape[0],
            axes=qd.static(eoa(sap_state.fem_vertices.v)),
        ):
            sap_state.fem_vertices.v[i_v, i_b] = fem_state.vertices.vel[i_step + 1, i_v, i_b]
    if qd.static(sap_config.has_rigid):
        for i_d, i_b in qd.ndrange(
            sap_info.n_dofs[None], sap_state.envs.is_active.shape[0], axes=qd.static(eoa(sap_state.rigid_dofs.v))
        ):
            sap_state.rigid_dofs.v[i_d, i_b] = dofs_state.vel[i_d, i_b]
    for i_b in range(sap_state.envs.is_active.shape[0]):
        sap_state.envs.is_active[i_b] = True


@qd.kernel
def kernel_init_fem_unconstrained_gradient_diag(
    i_step: int, fem_state: FEMState, sap_state: SAPState, fem_info: FEMInfo
):
    """Clear the gradient of the FEM vertices, and set the diagonal blocks of their Hessian to the elastic ones."""
    for i_v, i_b in qd.ndrange(
        sap_state.fem_vertices.v.shape[0],
        sap_state.envs.is_active.shape[0],
        axes=qd.static(eoa(sap_state.fem_vertices.gradient)),
    ):
        sap_state.fem_vertices.gradient[i_v, i_b] = qd.Vector.zero(gs.qd_float, 3)
        # The FEM solver solves for positions and SAP for velocities, so the Hessian scales by dt^2
        sap_state.fem_vertices.diag3x3[i_v, i_b] = fem_state.vertices_solve.diag3x3[i_v, i_b] * fem_info.dt2[None]
        sap_state.fem_vertices.v_diff[i_v, i_b] = (
            sap_state.fem_vertices.v[i_v, i_b] - fem_state.vertices.vel[i_step + 1, i_v, i_b]
        )


@qd.kernel
def kernel_init_rigid_unconstrained_gradient(dofs_state: DofsState, sap_state: SAPState, sap_info: SAPInfo):
    """Clear the gradient of the rigid dofs, and compute the difference of their velocities to the free-motion ones."""
    for i_d, i_b in qd.ndrange(
        sap_info.n_dofs[None], sap_state.envs.is_active.shape[0], axes=qd.static(eoa(sap_state.rigid_dofs.gradient))
    ):
        sap_state.rigid_dofs.gradient[i_d, i_b] = 0.0
        sap_state.rigid_dofs.v_diff[i_d, i_b] = sap_state.rigid_dofs.v[i_d, i_b] - dofs_state.vel[i_d, i_b]


@qd.kernel
def kernel_compute_fem_unconstrained_gradient(
    fem_state: FEMState,
    sap_state: SAPState,
    fem_info: FEMInfo,
    vertex_constraints: FEMVertexConstraints,
    fem_config: qd.template(),
):
    """Compute the gradient of the dynamic energy of the FEM vertices of the environments still solving."""
    func_compute_fem_matrix_vector_product(
        sap_state.fem_vertices.v_diff,
        sap_state.fem_vertices.gradient,
        sap_state.envs.is_active,
        fem_state,
        fem_info,
        vertex_constraints,
        fem_config,
    )


@qd.kernel
def kernel_compute_rigid_unconstrained_gradient(sap_state: SAPState, rigid_info: RigidInfo, sap_info: SAPInfo):
    """Compute the gradient of the dynamic energy of the rigid dofs of the environments still solving."""
    n_dofs = sap_info.n_dofs[None]
    for i_b, i_d0, i_d1 in qd.ndrange(sap_state.envs.is_active.shape[0], n_dofs, n_dofs):
        if not sap_state.envs.is_active[i_b]:
            continue
        qd.atomic_add(
            sap_state.rigid_dofs.gradient[i_d1, i_b],
            rigid_info.mass_mat[i_d1, i_d0, i_b] * sap_state.rigid_dofs.v_diff[i_d0, i_b],
        )


@qd.kernel
def kernel_compute_constraint_contact_gradient_hessian_diag_prec(
    sap_state: SAPState,
    fem_info: FEMInfo,
    vertex_constraints: FEMVertexConstraints,
    sap_info: SAPInfo,
    fem_config: qd.template(),
    sap_config: qd.template(),
    contact_handlers: qd.template(),
    constraint_handler: qd.template(),
):
    """Add the impulses of the constraints and contacts to the gradient and their Hessian to its diagonal blocks.

    The inverse of the diagonal blocks of the FEM vertices is the preconditioner of the preconditioned conjugate
    gradient (PCG) solve.
    """
    if qd.static(sap_config.has_fem):
        for i_v, i_b in qd.ndrange(
            sap_state.fem_vertices.v.shape[0],
            sap_state.envs.is_active.shape[0],
            axes=qd.static(eoa(sap_state.fem_vertices.impulse)),
        ):
            if not sap_state.envs.is_active[i_b]:
                continue
            sap_state.fem_vertices.impulse[i_v, i_b] = qd.Vector.zero(gs.qd_float, 3)
    if qd.static(sap_config.has_rigid):
        for i_d, i_b in qd.ndrange(
            sap_info.n_dofs[None], sap_state.envs.is_active.shape[0], axes=qd.static(eoa(sap_state.rigid_dofs.impulse))
        ):
            if not sap_state.envs.is_active[i_b]:
                continue
            sap_state.rigid_dofs.impulse[i_d, i_b] = 0.0
    if qd.static(sap_config.has_equality_constraints):
        constraint_handler.compute_gradient_hessian_diag(sap_state)
    for contact in qd.static(contact_handlers):
        contact.compute_gradient_hessian_diag(sap_state, fem_info, vertex_constraints, sap_info, fem_config)
    if qd.static(sap_config.has_fem):
        for i_v, i_b in qd.ndrange(
            sap_state.fem_vertices.v.shape[0],
            sap_state.envs.is_active.shape[0],
            axes=qd.static(eoa(sap_state.fem_vertices.prec)),
        ):
            if not sap_state.envs.is_active[i_b]:
                continue
            sap_state.fem_vertices.prec[i_v, i_b] = sap_state.fem_vertices.diag3x3[i_v, i_b].inverse()


@qd.kernel
def kernel_check_sap_convergence(
    sap_state: SAPState, fem_info: FEMInfo, rigid_info: RigidInfo, sap_info: SAPInfo, sap_config: qd.template()
):
    """Stop the SAP solve of the environments whose gradient is small against their momentum and impulse."""
    n_envs = sap_state.envs.is_active.shape[0]
    for i_b in range(n_envs):
        if not sap_state.envs.is_active[i_b]:
            continue
        sap_state.envs.gradient_norm[i_b] = 0.0
        sap_state.envs.momentum_norm[i_b] = 0.0
        sap_state.envs.impulse_norm[i_b] = 0.0
    if qd.static(sap_config.has_fem):
        for i_v, i_b in qd.ndrange(
            sap_state.fem_vertices.v.shape[0], n_envs, axes=qd.static(eoa(sap_state.fem_vertices.gradient))
        ):
            if not sap_state.envs.is_active[i_b]:
                continue
            qd.atomic_add(
                sap_state.envs.gradient_norm[i_b],
                sap_state.fem_vertices.gradient[i_v, i_b].norm_sqr() / fem_info.vertices.mass[i_v],
            )
            qd.atomic_add(
                sap_state.envs.momentum_norm[i_b],
                sap_state.fem_vertices.v[i_v, i_b].norm_sqr() * fem_info.vertices.mass[i_v],
            )
            qd.atomic_add(
                sap_state.envs.impulse_norm[i_b],
                sap_state.fem_vertices.impulse[i_v, i_b].norm_sqr() / fem_info.vertices.mass[i_v],
            )
    if qd.static(sap_config.has_rigid):
        for i_d, i_b in qd.ndrange(sap_info.n_dofs[None], n_envs, axes=qd.static(eoa(rigid_info.mass_mat))):
            if not sap_state.envs.is_active[i_b]:
                continue
            qd.atomic_add(
                sap_state.envs.gradient_norm[i_b],
                sap_state.rigid_dofs.gradient[i_d, i_b] ** 2 / rigid_info.mass_mat[i_d, i_d, i_b],
            )
            qd.atomic_add(
                sap_state.envs.momentum_norm[i_b],
                sap_state.rigid_dofs.v[i_d, i_b] ** 2 * rigid_info.mass_mat[i_d, i_d, i_b],
            )
            qd.atomic_add(
                sap_state.envs.impulse_norm[i_b],
                sap_state.rigid_dofs.impulse[i_d, i_b] ** 2 / rigid_info.mass_mat[i_d, i_d, i_b],
            )
    for i_b in range(n_envs):
        if not sap_state.envs.is_active[i_b]:
            continue
        norm_thr = sap_info.convergence_atol[None] + sap_info.convergence_rtol[None] * qd.max(
            sap_state.envs.momentum_norm[i_b], sap_state.envs.impulse_norm[i_b]
        )
        sap_state.envs.is_active[i_b] = sap_state.envs.gradient_norm[i_b] >= norm_thr


@qd.kernel
def kernel_init_pcg_solve(
    sap_state: SAPState,
    entities_info: EntitiesInfo,
    rigid_info: RigidInfo,
    sap_info: SAPInfo,
    sap_config: qd.template(),
):
    """Start the preconditioned conjugate gradient (PCG) solve of the search direction from zero."""
    n_envs = sap_state.envs.is_active.shape[0]
    for i_b in range(n_envs):
        sap_state.envs.is_pcg_active[i_b] = sap_state.envs.is_active[i_b]
        if not sap_state.envs.is_pcg_active[i_b]:
            continue
        sap_state.envs.rTr[i_b] = 0.0
        sap_state.envs.rTz[i_b] = 0.0
    if qd.static(sap_config.has_fem):
        for i_v, i_b in qd.ndrange(
            sap_state.fem_vertices.v.shape[0], n_envs, axes=qd.static(eoa(sap_state.fem_vertices.x))
        ):
            if not sap_state.envs.is_pcg_active[i_b]:
                continue
            sap_state.fem_vertices.x[i_v, i_b] = qd.Vector.zero(gs.qd_float, 3)
            sap_state.fem_vertices.r[i_v, i_b] = -sap_state.fem_vertices.gradient[i_v, i_b]
            sap_state.fem_vertices.z[i_v, i_b] = (
                sap_state.fem_vertices.prec[i_v, i_b] @ sap_state.fem_vertices.r[i_v, i_b]
            )
            sap_state.fem_vertices.p[i_v, i_b] = sap_state.fem_vertices.z[i_v, i_b]
            qd.atomic_add(
                sap_state.envs.rTr[i_b], sap_state.fem_vertices.r[i_v, i_b].dot(sap_state.fem_vertices.r[i_v, i_b])
            )
            qd.atomic_add(
                sap_state.envs.rTz[i_b], sap_state.fem_vertices.r[i_v, i_b].dot(sap_state.fem_vertices.z[i_v, i_b])
            )
    if qd.static(sap_config.has_rigid):
        for i_d, i_b in qd.ndrange(sap_info.n_dofs[None], n_envs, axes=qd.static(eoa(sap_state.rigid_dofs.x))):
            if not sap_state.envs.is_pcg_active[i_b]:
                continue
            sap_state.rigid_dofs.x[i_d, i_b] = 0.0
            sap_state.rigid_dofs.r[i_d, i_b] = -sap_state.rigid_dofs.gradient[i_d, i_b]
            qd.atomic_add(sap_state.envs.rTr[i_b], sap_state.rigid_dofs.r[i_d, i_b] ** 2)

        func_rigid_solve_pcg(
            sap_state.rigid_dofs.r,
            sap_state.rigid_dofs.z,
            sap_state.envs.is_pcg_active,
            entities_info,
            rigid_info,
            sap_info,
        )

        for i_d, i_b in qd.ndrange(sap_info.n_dofs[None], n_envs, axes=qd.static(eoa(sap_state.rigid_dofs.p))):
            if not sap_state.envs.is_pcg_active[i_b]:
                continue
            sap_state.rigid_dofs.p[i_d, i_b] = sap_state.rigid_dofs.z[i_d, i_b]
            qd.atomic_add(sap_state.envs.rTz[i_b], sap_state.rigid_dofs.r[i_d, i_b] * sap_state.rigid_dofs.z[i_d, i_b])
    for i_b in range(n_envs):
        if not sap_state.envs.is_pcg_active[i_b]:
            continue
        sap_state.envs.is_pcg_active[i_b] = sap_state.envs.rTr[i_b] > sap_info.pcg_threshold[None]


@qd.kernel
def kernel_one_pcg_iter(
    fem_state: FEMState,
    sap_state: SAPState,
    fem_info: FEMInfo,
    vertex_constraints: FEMVertexConstraints,
    entities_info: EntitiesInfo,
    rigid_info: RigidInfo,
    sap_info: SAPInfo,
    fem_config: qd.template(),
    sap_config: qd.template(),
    contact_handlers: qd.template(),
    constraint_handler: qd.template(),
):
    """Run one iteration of the solve of the search direction of the environments still solving.

    The solve is a preconditioned conjugate gradient (PCG).

    See https://en.wikipedia.org/wiki/Conjugate_gradient_method#The_preconditioned_conjugate_gradient_method
    """
    n_envs = sap_state.envs.is_active.shape[0]
    n_dofs = sap_info.n_dofs[None]

    # The product Ap of the Hessian with the search direction
    if qd.static(sap_config.has_fem):
        func_compute_fem_matrix_vector_product(
            sap_state.fem_vertices.p,
            sap_state.fem_vertices.Ap,
            sap_state.envs.is_pcg_active,
            fem_state,
            fem_info,
            vertex_constraints,
            fem_config,
        )
    if qd.static(sap_config.has_rigid):
        func_compute_rigid_mass_mat_vec_product(
            sap_state.rigid_dofs.p, sap_state.rigid_dofs.Ap, sap_state.envs.is_pcg_active, rigid_info, sap_info
        )
    if qd.static(sap_config.has_equality_constraints):
        constraint_handler.compute_Ap(sap_state)
    for contact in qd.static(contact_handlers):
        contact.compute_pcg_matrix_vector_product(sap_state, fem_info, vertex_constraints, sap_info, fem_config)

    for i_b in range(n_envs):
        if not sap_state.envs.is_pcg_active[i_b]:
            continue
        sap_state.envs.pTAp[i_b] = 0.0
        sap_state.envs.rTr_new[i_b] = 0.0
        sap_state.envs.rTz_new[i_b] = 0.0

    if qd.static(sap_config.has_fem):
        for i_v, i_b in qd.ndrange(
            sap_state.fem_vertices.v.shape[0], n_envs, axes=qd.static(eoa(sap_state.fem_vertices.Ap))
        ):
            if not sap_state.envs.is_pcg_active[i_b]:
                continue
            qd.atomic_add(
                sap_state.envs.pTAp[i_b], sap_state.fem_vertices.p[i_v, i_b].dot(sap_state.fem_vertices.Ap[i_v, i_b])
            )
    if qd.static(sap_config.has_rigid):
        for i_d, i_b in qd.ndrange(n_dofs, n_envs, axes=qd.static(eoa(sap_state.rigid_dofs.p))):
            if not sap_state.envs.is_pcg_active[i_b]:
                continue
            qd.atomic_add(
                sap_state.envs.pTAp[i_b], sap_state.rigid_dofs.p[i_d, i_b] * sap_state.rigid_dofs.Ap[i_d, i_b]
            )

    for i_b in range(n_envs):
        if not sap_state.envs.is_pcg_active[i_b]:
            continue
        sap_state.envs.alpha[i_b] = sap_state.envs.rTz[i_b] / sap_state.envs.pTAp[i_b]

    if qd.static(sap_config.has_fem):
        for i_v, i_b in qd.ndrange(
            sap_state.fem_vertices.v.shape[0], n_envs, axes=qd.static(eoa(sap_state.fem_vertices.x))
        ):
            if not sap_state.envs.is_pcg_active[i_b]:
                continue
            sap_state.fem_vertices.x[i_v, i_b] = (
                sap_state.fem_vertices.x[i_v, i_b] + sap_state.envs.alpha[i_b] * sap_state.fem_vertices.p[i_v, i_b]
            )
            sap_state.fem_vertices.r[i_v, i_b] = (
                sap_state.fem_vertices.r[i_v, i_b] - sap_state.envs.alpha[i_b] * sap_state.fem_vertices.Ap[i_v, i_b]
            )
            sap_state.fem_vertices.z[i_v, i_b] = (
                sap_state.fem_vertices.prec[i_v, i_b] @ sap_state.fem_vertices.r[i_v, i_b]
            )
            qd.atomic_add(sap_state.envs.rTr_new[i_b], sap_state.fem_vertices.r[i_v, i_b].norm_sqr())
            qd.atomic_add(
                sap_state.envs.rTz_new[i_b], sap_state.fem_vertices.r[i_v, i_b].dot(sap_state.fem_vertices.z[i_v, i_b])
            )
    if qd.static(sap_config.has_rigid):
        for i_d, i_b in qd.ndrange(n_dofs, n_envs, axes=qd.static(eoa(sap_state.rigid_dofs.x))):
            if not sap_state.envs.is_pcg_active[i_b]:
                continue
            sap_state.rigid_dofs.x[i_d, i_b] = (
                sap_state.rigid_dofs.x[i_d, i_b] + sap_state.envs.alpha[i_b] * sap_state.rigid_dofs.p[i_d, i_b]
            )
            sap_state.rigid_dofs.r[i_d, i_b] = (
                sap_state.rigid_dofs.r[i_d, i_b] - sap_state.envs.alpha[i_b] * sap_state.rigid_dofs.Ap[i_d, i_b]
            )
            qd.atomic_add(
                sap_state.envs.rTr_new[i_b], sap_state.rigid_dofs.r[i_d, i_b] * sap_state.rigid_dofs.r[i_d, i_b]
            )

        func_rigid_solve_pcg(
            sap_state.rigid_dofs.r,
            sap_state.rigid_dofs.z,
            sap_state.envs.is_pcg_active,
            entities_info,
            rigid_info,
            sap_info,
        )

        for i_d, i_b in qd.ndrange(n_dofs, n_envs, axes=qd.static(eoa(sap_state.rigid_dofs.r))):
            if not sap_state.envs.is_pcg_active[i_b]:
                continue
            qd.atomic_add(
                sap_state.envs.rTz_new[i_b], sap_state.rigid_dofs.r[i_d, i_b] * sap_state.rigid_dofs.z[i_d, i_b]
            )

    for i_b in range(n_envs):
        if not sap_state.envs.is_pcg_active[i_b]:
            continue
        sap_state.envs.is_pcg_active[i_b] = sap_state.envs.rTr_new[i_b] > sap_info.pcg_threshold[None]
    for i_b in range(n_envs):
        if not sap_state.envs.is_pcg_active[i_b]:
            continue
        sap_state.envs.beta[i_b] = sap_state.envs.rTz_new[i_b] / sap_state.envs.rTz[i_b]
        sap_state.envs.rTr[i_b] = sap_state.envs.rTr_new[i_b]
        sap_state.envs.rTz[i_b] = sap_state.envs.rTz_new[i_b]

    if qd.static(sap_config.has_fem):
        for i_v, i_b in qd.ndrange(
            sap_state.fem_vertices.v.shape[0], n_envs, axes=qd.static(eoa(sap_state.fem_vertices.p))
        ):
            if not sap_state.envs.is_pcg_active[i_b]:
                continue
            sap_state.fem_vertices.p[i_v, i_b] = (
                sap_state.fem_vertices.z[i_v, i_b] + sap_state.envs.beta[i_b] * sap_state.fem_vertices.p[i_v, i_b]
            )
    if qd.static(sap_config.has_rigid):
        for i_d, i_b in qd.ndrange(n_dofs, n_envs, axes=qd.static(eoa(sap_state.rigid_dofs.p))):
            if not sap_state.envs.is_pcg_active[i_b]:
                continue
            sap_state.rigid_dofs.p[i_d, i_b] = (
                sap_state.rigid_dofs.z[i_d, i_b] + sap_state.envs.beta[i_b] * sap_state.rigid_dofs.p[i_d, i_b]
            )


@qd.kernel
def kernel_init_exact_linesearch(
    i_step: int,
    fem_state: FEMState,
    dofs_state: DofsState,
    sap_state: SAPState,
    fem_info: FEMInfo,
    vertex_constraints: FEMVertexConstraints,
    rigid_info: RigidInfo,
    sap_info: SAPInfo,
    fem_config: qd.template(),
    sap_config: qd.template(),
    contact_handlers: qd.template(),
    constraint_handler: qd.template(),
):
    """Start the exact line search along the search direction with a Newton step, bracketed by the maximum step."""
    n_envs = sap_state.envs.is_active.shape[0]
    n_dofs = sap_info.n_dofs[None]

    for i_b in range(n_envs):
        sap_state.envs.is_linesearch_active[i_b] = sap_state.envs.is_active[i_b]
        if not sap_state.envs.is_linesearch_active[i_b]:
            continue
        sap_state.envs.step_size[i_b] = sap_info.linesearch_max_step_size[None]
        sap_state.envs.m[i_b] = 0.0
    if qd.static(sap_config.has_fem):
        for i_v, i_b in qd.ndrange(
            sap_state.fem_vertices.v.shape[0], n_envs, axes=qd.static(eoa(sap_state.fem_vertices.x_prev))
        ):
            if not sap_state.envs.is_linesearch_active[i_b]:
                continue
            qd.atomic_add(
                sap_state.envs.m[i_b], sap_state.fem_vertices.x[i_v, i_b].dot(sap_state.fem_vertices.gradient[i_v, i_b])
            )
            sap_state.fem_vertices.x_prev[i_v, i_b] = sap_state.fem_vertices.v[i_v, i_b]
    if qd.static(sap_config.has_rigid):
        for i_d, i_b in qd.ndrange(n_dofs, n_envs, axes=qd.static(eoa(sap_state.rigid_dofs.x_prev))):
            if not sap_state.envs.is_linesearch_active[i_b]:
                continue
            qd.atomic_add(
                sap_state.envs.m[i_b], sap_state.rigid_dofs.x[i_d, i_b] * sap_state.rigid_dofs.gradient[i_d, i_b]
            )
            sap_state.rigid_dofs.x_prev[i_d, i_b] = sap_state.rigid_dofs.v[i_d, i_b]

    # The total energy at the start of the line search
    for i_b in range(n_envs):
        sap_state.envs.prev_energy[i_b] = 0.0
    if qd.static(sap_config.has_fem):
        # Inertia
        for i_v, i_b in qd.ndrange(
            sap_state.fem_vertices.v.shape[0], n_envs, axes=qd.static(eoa(sap_state.fem_vertices.v_diff))
        ):
            if not sap_state.envs.is_linesearch_active[i_b]:
                continue
            sap_state.fem_vertices.v_diff[i_v, i_b] = (
                sap_state.fem_vertices.v[i_v, i_b] - fem_state.vertices.vel[i_step + 1, i_v, i_b]
            )
            qd.atomic_add(
                sap_state.envs.prev_energy[i_b],
                0.5
                * fem_info.vertices.mass_over_dt2[i_v]
                * sap_state.fem_vertices.v_diff[i_v, i_b].norm_sqr()
                * fem_info.dt2[None]
                * (fem_info.damping_alpha_dt[None] + 1.0),
            )
        # Elasticity
        for i_e, i_b in qd.ndrange(
            fem_info.elements.V.shape[0], n_envs, axes=qd.static(eoa(sap_state.fem_vertices.v_diff))
        ):
            if not sap_state.envs.is_linesearch_active[i_b]:
                continue
            V_dt2 = fem_info.elements.V[i_e] * fem_info.dt2[None]
            S = func_element_mapping_matrix(i_e, i_b, fem_info, vertex_constraints, fem_config)
            i_vs = fem_info.elements.el2v[i_e]
            x = qd.Matrix.cols([sap_state.fem_vertices.v_diff[i_vs[i], i_b] for i in qd.static(range(4))])
            dF, dP = func_elastic_product(i_e, i_b, S, x, fem_state, fem_info, fem_config)
            qd.atomic_add(
                sap_state.envs.prev_energy[i_b], 0.5 * (dF * dP).sum() * fem_info.damping_beta_factor[None] * V_dt2
            )
    if qd.static(sap_config.has_rigid):
        # Kinetic energy
        for i_d, i_b in qd.ndrange(n_dofs, n_envs, axes=qd.static(eoa(sap_state.rigid_dofs.v_diff))):
            if not sap_state.envs.is_linesearch_active[i_b]:
                continue
            sap_state.rigid_dofs.v_diff[i_d, i_b] = sap_state.rigid_dofs.v[i_d, i_b] - dofs_state.vel[i_d, i_b]
        func_compute_rigid_mass_mat_vec_product(
            sap_state.rigid_dofs.v_diff,
            sap_state.rigid_dofs.mass_v_diff,
            sap_state.envs.is_linesearch_active,
            rigid_info,
            sap_info,
        )
        for i_d, i_b in qd.ndrange(n_dofs, n_envs, axes=qd.static(eoa(sap_state.rigid_dofs.mass_v_diff))):
            if not sap_state.envs.is_linesearch_active[i_b]:
                continue
            qd.atomic_add(
                sap_state.envs.prev_energy[i_b],
                0.5 * sap_state.rigid_dofs.v_diff[i_d, i_b] * sap_state.rigid_dofs.mass_v_diff[i_d, i_b],
            )
    if qd.static(sap_config.has_equality_constraints):
        constraint_handler.compute_energy(sap_state)
    for contact in qd.static(contact_handlers):
        contact.compute_energy(sap_state, fem_info, sap_info)

    # The product of the Hessian with the search direction, and its second derivative along the step size
    if qd.static(sap_config.has_fem):
        func_compute_fem_matrix_vector_product(
            sap_state.fem_vertices.x,
            sap_state.fem_vertices.dp,
            sap_state.envs.is_linesearch_active,
            fem_state,
            fem_info,
            vertex_constraints,
            fem_config,
        )
    if qd.static(sap_config.has_rigid):
        func_compute_rigid_mass_mat_vec_product(
            sap_state.rigid_dofs.x, sap_state.rigid_dofs.dp, sap_state.envs.is_linesearch_active, rigid_info, sap_info
        )
    if qd.static(sap_config.has_equality_constraints):
        constraint_handler.prepare_search_direction_data(sap_state)
    for contact in qd.static(contact_handlers):
        contact.prepare_search_direction_data(sap_state, fem_info, sap_info)
    for i_b in range(n_envs):
        sap_state.envs.d2ellA_dalpha2[i_b] = 0.0
    if qd.static(sap_config.has_fem):
        for i_v, i_b in qd.ndrange(
            sap_state.fem_vertices.v.shape[0], n_envs, axes=qd.static(eoa(sap_state.fem_vertices.dp))
        ):
            if not sap_state.envs.is_linesearch_active[i_b]:
                continue
            qd.atomic_add(
                sap_state.envs.d2ellA_dalpha2[i_b],
                sap_state.fem_vertices.x[i_v, i_b].dot(sap_state.fem_vertices.dp[i_v, i_b]),
            )
    if qd.static(sap_config.has_rigid):
        for i_d, i_b in qd.ndrange(n_dofs, n_envs, axes=qd.static(eoa(sap_state.rigid_dofs.x))):
            if not sap_state.envs.is_linesearch_active[i_b]:
                continue
            qd.atomic_add(
                sap_state.envs.d2ellA_dalpha2[i_b], sap_state.rigid_dofs.x[i_d, i_b] * sap_state.rigid_dofs.dp[i_d, i_b]
            )

    func_update_velocity_linesearch(sap_state, sap_info, sap_config)
    func_compute_line_energy_gradient_hessian(
        i_step,
        fem_state,
        dofs_state,
        sap_state,
        fem_info,
        sap_info,
        sap_config,
        contact_handlers,
        constraint_handler,
    )

    # The line search stops at once for an energy increasing along the search direction
    for i_b in range(n_envs):
        if not sap_state.envs.is_linesearch_active[i_b]:
            continue
        sap_state.envs.is_linesearch_active[i_b] = sap_state.envs.dell_dalpha[i_b] > 0.0
    if qd.static(sap_config.has_fem):
        for i_v, i_b in qd.ndrange(
            sap_state.fem_vertices.v.shape[0], n_envs, axes=qd.static(eoa(sap_state.fem_vertices.v))
        ):
            if not sap_state.envs.is_linesearch_active[i_b]:
                continue
            err_threshold = (
                sap_info.convergence_atol[None] + sap_info.convergence_rtol[None] * sap_state.envs.prev_energy[i_b]
            )
            if -sap_state.envs.m[i_b] < err_threshold:
                sap_state.fem_vertices.v[i_v, i_b] = (
                    sap_state.fem_vertices.x_prev[i_v, i_b] + sap_state.fem_vertices.x[i_v, i_b]
                )
    if qd.static(sap_config.has_rigid):
        for i_d, i_b in qd.ndrange(n_dofs, n_envs, axes=qd.static(eoa(sap_state.rigid_dofs.v))):
            if not sap_state.envs.is_linesearch_active[i_b]:
                continue
            err_threshold = (
                sap_info.convergence_atol[None] + sap_info.convergence_rtol[None] * sap_state.envs.prev_energy[i_b]
            )
            if -sap_state.envs.m[i_b] < err_threshold:
                sap_state.rigid_dofs.v[i_d, i_b] = (
                    sap_state.rigid_dofs.x_prev[i_d, i_b] + sap_state.rigid_dofs.x[i_d, i_b]
                )
    # A derivative of the cost along the search direction already below the tolerance at the full step takes that
    # step and ends the line search, a rare case of the safeguarded Newton line search (rtsafe).
    for i_b in range(n_envs):
        if not sap_state.envs.is_linesearch_active[i_b]:
            continue
        err_threshold = (
            sap_info.convergence_atol[None] + sap_info.convergence_rtol[None] * sap_state.envs.prev_energy[i_b]
        )
        if -sap_state.envs.m[i_b] < err_threshold:
            sap_state.envs.is_linesearch_active[i_b] = False
            sap_state.envs.step_size[i_b] = 1.0

    for i_b in range(n_envs):
        if not sap_state.envs.is_linesearch_active[i_b]:
            continue
        sap_state.envs.dell_scale[i_b] = -sap_state.envs.m[i_b]
        sap_state.envs.step_size[i_b] = qd.min(
            -sap_state.envs.m[i_b] / sap_state.envs.d2ell_dalpha2[i_b], sap_info.linesearch_max_step_size[None]
        )
        sap_state.envs.alpha_min[i_b] = 0.0
        sap_state.envs.alpha_max[i_b] = sap_info.linesearch_max_step_size[None]
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


@qd.kernel
def kernel_one_exact_linesearch_iter(
    i_step: int,
    fem_state: FEMState,
    dofs_state: DofsState,
    sap_state: SAPState,
    fem_info: FEMInfo,
    sap_info: SAPInfo,
    sap_config: qd.template(),
    contact_handlers: qd.template(),
    constraint_handler: qd.template(),
):
    """Run one iteration of the exact line search, a Newton step falling back to bisection out of the bracket.

    The bracket [alpha_min, alpha_max] holds the root of the derivative of the total energy along the step size.
    """
    func_update_velocity_linesearch(sap_state, sap_info, sap_config)
    func_compute_line_energy_gradient_hessian(
        i_step,
        fem_state,
        dofs_state,
        sap_state,
        fem_info,
        sap_info,
        sap_config,
        contact_handlers,
        constraint_handler,
    )

    n_envs = sap_state.envs.is_active.shape[0]
    for i_b in range(n_envs):
        if not sap_state.envs.is_linesearch_active[i_b]:
            continue
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

    for i_b in range(n_envs):
        if not sap_state.envs.is_linesearch_active[i_b]:
            continue
        is_newton_slow = 2.0 * qd.abs(sap_state.envs.f[i_b]) > qd.abs(
            sap_state.envs.minus_dalpha_prev[i_b] * sap_state.envs.df[i_b]
        )
        sap_state.envs.minus_dalpha_prev[i_b] = sap_state.envs.minus_dalpha[i_b]
        if is_newton_slow:
            # bisect
            sap_state.envs.minus_dalpha[i_b] = 0.5 * (sap_state.envs.alpha_min[i_b] - sap_state.envs.alpha_max[i_b])
            sap_state.envs.step_size[i_b] = sap_state.envs.alpha_min[i_b] - sap_state.envs.minus_dalpha[i_b]
        else:
            # newton
            sap_state.envs.minus_dalpha[i_b] = sap_state.envs.f[i_b] / sap_state.envs.df[i_b]
            sap_state.envs.step_size[i_b] = sap_state.envs.step_size[i_b] - sap_state.envs.minus_dalpha[i_b]
            if (
                sap_state.envs.step_size[i_b] <= sap_state.envs.alpha_min[i_b]
                or sap_state.envs.step_size[i_b] >= sap_state.envs.alpha_max[i_b]
            ):
                # bisect
                sap_state.envs.minus_dalpha[i_b] = 0.5 * (sap_state.envs.alpha_min[i_b] - sap_state.envs.alpha_max[i_b])
                sap_state.envs.step_size[i_b] = sap_state.envs.alpha_min[i_b] - sap_state.envs.minus_dalpha[i_b]
        if qd.abs(sap_state.envs.minus_dalpha[i_b]) < sap_state.envs.alpha_tol[i_b]:
            sap_state.envs.is_linesearch_active[i_b] = False


@qd.func
def func_compute_fem_matrix_vector_product(
    src: qd.Tensor,
    dst: qd.Tensor,
    is_active: qd.Tensor,
    fem_state: FEMState,
    fem_info: FEMInfo,
    vertex_constraints: FEMVertexConstraints,
    fem_config: qd.template(),
):
    """Compute the product of the FEM Hessian, mass matrix and elastic stiffness, with a vector of the vertices."""
    # Inertia
    for i_v, i_b in qd.ndrange(*dst.shape, axes=qd.static(eoa(dst))):
        if not is_active[i_b]:
            continue
        dst[i_v, i_b] = (
            fem_info.vertices.mass_over_dt2[i_v]
            * src[i_v, i_b]
            * fem_info.dt2[None]
            * (fem_info.damping_alpha_dt[None] + 1.0)
        )

    # Elasticity
    for i_e, i_b in qd.ndrange(fem_info.elements.V.shape[0], dst.shape[1], axes=qd.static(eoa(dst))):
        if not is_active[i_b]:
            continue
        V_dt2 = fem_info.elements.V[i_e] * fem_info.dt2[None]
        S = func_element_mapping_matrix(i_e, i_b, fem_info, vertex_constraints, fem_config)
        i_vs = fem_info.elements.el2v[i_e]
        x = qd.Matrix.cols([src[i_vs[i], i_b] for i in qd.static(range(4))])
        _, dP = func_elastic_product(i_e, i_b, S, x, fem_state, fem_info, fem_config)
        A_src = V_dt2 * fem_info.damping_beta_factor[None] * dP @ S.transpose()
        for i in qd.static(range(4)):
            qd.atomic_add(dst[i_vs[i], i_b], A_src[:, i])


@qd.func
def func_compute_rigid_mass_mat_vec_product(
    vec: qd.Tensor, out: qd.Tensor, is_active: qd.Tensor, rigid_info: RigidInfo, sap_info: SAPInfo
):
    """Compute the product of the rigid mass matrix with a vector of the dofs."""
    for i_d1, i_b in qd.ndrange(sap_info.n_dofs[None], out.shape[1], axes=qd.static(eoa(rigid_info.mass_mat))):
        prod = gs.qd_float(0.0)
        if is_active[i_b]:
            for i_d0 in range(sap_info.n_dofs[None]):
                prod = prod + rigid_info.mass_mat[i_d1, i_d0, i_b] * vec[i_d0, i_b]
        out[i_d1, i_b] = prod


# FIXME: This following two rigid solves are duplicated with 'func_solve_mass_batch' in rigid/abd/forward_dynamics.py.
# Consider refactoring.
@qd.func
def func_rigid_solve_pcg(
    vec: qd.Tensor,
    out: qd.Tensor,
    is_active: qd.Tensor,
    entities_info: EntitiesInfo,
    rigid_info: RigidInfo,
    sap_info: SAPInfo,
):
    # Step 1: Solve w st. L^T @ w = y
    for i_e, i_b in qd.ndrange(
        entities_info.dof_start.shape[0], out.shape[1], axes=qd.static(eoa(rigid_info.mass_mat_L))
    ):
        if not is_active[i_b]:
            continue
        entity_dof_end = entities_info.dof_end[i_e]
        n_dofs = entities_info.n_dofs[i_e]
        for i_d_ in range(n_dofs):
            i_d = entity_dof_end - i_d_ - 1
            out[i_d, i_b] = vec[i_d, i_b]
            for j_d in range(i_d + 1, entity_dof_end):
                out[i_d, i_b] = out[i_d, i_b] - rigid_info.mass_mat_L[j_d, i_d, i_b] * out[j_d, i_b]

    # Step 2: z = D^{-1} w
    for i_d, i_b in qd.ndrange(sap_info.n_dofs[None], out.shape[1], axes=qd.static(eoa(rigid_info.mass_mat_D_inv))):
        if not is_active[i_b]:
            continue
        out[i_d, i_b] = out[i_d, i_b] * rigid_info.mass_mat_D_inv[i_d, i_b]

    # Step 3: Solve x st. L @ x = z
    for i_e, i_b in qd.ndrange(
        entities_info.dof_start.shape[0], out.shape[1], axes=qd.static(eoa(rigid_info.mass_mat_L))
    ):
        if not is_active[i_b]:
            continue
        entity_dof_start = entities_info.dof_start[i_e]
        entity_dof_end = entities_info.dof_end[i_e]
        for i_d in range(entity_dof_start, entity_dof_end):
            for j_d in range(entity_dof_start, i_d):
                out[i_d, i_b] = out[i_d, i_b] - rigid_info.mass_mat_L[i_d, j_d, i_b] * out[j_d, i_b]


@qd.func
def func_rigid_solve_jacobian(
    n_pairs: int,
    pairs_env_idx: qd.Tensor,
    vec: qd.Tensor,
    out: qd.Tensor,
    entities_info: EntitiesInfo,
    rigid_info: RigidInfo,
    sap_info: SAPInfo,
    dim: int,
):
    """Solve the rigid mass matrix against the transposed Jacobian of every contact or constraint, row by row."""
    # Step 1: Solve w st. L^T @ w = y
    for i_p, i_e, k in qd.ndrange(n_pairs, entities_info.dof_start.shape[0], dim):
        i_b = pairs_env_idx[i_p]
        entity_dof_end = entities_info.dof_end[i_e]
        n_dofs = entities_info.n_dofs[i_e]
        for i_d_ in range(n_dofs):
            i_d = entity_dof_end - i_d_ - 1
            out[i_p, i_d][k] = vec[i_p, i_d][k]
            for j_d in range(i_d + 1, entity_dof_end):
                out[i_p, i_d][k] = out[i_p, i_d][k] - rigid_info.mass_mat_L[j_d, i_d, i_b] * out[i_p, j_d][k]

    # Step 2: z = D^{-1} w
    for i_p, i_d, k in qd.ndrange(n_pairs, sap_info.n_dofs[None], dim):
        i_b = pairs_env_idx[i_p]
        out[i_p, i_d][k] = out[i_p, i_d][k] * rigid_info.mass_mat_D_inv[i_d, i_b]

    # Step 3: Solve x st. L @ x = z
    for i_p, i_e, k in qd.ndrange(n_pairs, entities_info.dof_start.shape[0], dim):
        i_b = pairs_env_idx[i_p]
        entity_dof_start = entities_info.dof_start[i_e]
        entity_dof_end = entities_info.dof_end[i_e]
        for i_d in range(entity_dof_start, entity_dof_end):
            for j_d in range(entity_dof_start, i_d):
                out[i_p, i_d][k] = out[i_p, i_d][k] - rigid_info.mass_mat_L[i_d, j_d, i_b] * out[i_p, j_d][k]


@qd.func
def func_update_velocity_linesearch(sap_state: SAPState, sap_info: SAPInfo, sap_config: qd.template()):
    """Move the velocities of the environments still searching to the current step size along the search direction."""
    if qd.static(sap_config.has_fem):
        for i_v, i_b in qd.ndrange(
            sap_state.fem_vertices.v.shape[0],
            sap_state.envs.is_active.shape[0],
            axes=qd.static(eoa(sap_state.fem_vertices.v)),
        ):
            if not sap_state.envs.is_linesearch_active[i_b]:
                continue
            sap_state.fem_vertices.v[i_v, i_b] = (
                sap_state.fem_vertices.x_prev[i_v, i_b]
                + sap_state.envs.step_size[i_b] * sap_state.fem_vertices.x[i_v, i_b]
            )
    if qd.static(sap_config.has_rigid):
        for i_d, i_b in qd.ndrange(
            sap_info.n_dofs[None], sap_state.envs.is_active.shape[0], axes=qd.static(eoa(sap_state.rigid_dofs.v))
        ):
            if not sap_state.envs.is_linesearch_active[i_b]:
                continue
            sap_state.rigid_dofs.v[i_d, i_b] = (
                sap_state.rigid_dofs.x_prev[i_d, i_b] + sap_state.envs.step_size[i_b] * sap_state.rigid_dofs.x[i_d, i_b]
            )


@qd.func
def func_compute_line_energy_gradient_hessian(
    i_step: int,
    fem_state: FEMState,
    dofs_state: DofsState,
    sap_state: SAPState,
    fem_info: FEMInfo,
    sap_info: SAPInfo,
    sap_config: qd.template(),
    contact_handlers: qd.template(),
    constraint_handler: qd.template(),
):
    """Compute the total energy at the current step size, and its first and second derivatives along the step size."""
    n_envs = sap_state.envs.is_active.shape[0]
    for i_b in range(n_envs):
        if not sap_state.envs.is_linesearch_active[i_b]:
            continue
        sap_state.envs.energy[i_b] = (
            sap_state.envs.prev_energy[i_b]
            + 0.5 * sap_state.envs.step_size[i_b] ** 2 * sap_state.envs.d2ellA_dalpha2[i_b]
        )
        sap_state.envs.dell_dalpha[i_b] = 0.0
        sap_state.envs.d2ell_dalpha2[i_b] = sap_state.envs.d2ellA_dalpha2[i_b]

    if qd.static(sap_config.has_fem):
        for i_v, i_b in qd.ndrange(
            sap_state.fem_vertices.v.shape[0], n_envs, axes=qd.static(eoa(sap_state.fem_vertices.dp))
        ):
            if not sap_state.envs.is_linesearch_active[i_b]:
                continue
            qd.atomic_add(
                sap_state.envs.energy[i_b],
                sap_state.envs.step_size[i_b]
                * sap_state.fem_vertices.dp[i_v, i_b].dot(
                    sap_state.fem_vertices.v[i_v, i_b] - fem_state.vertices.vel[i_step + 1, i_v, i_b]
                ),
            )
        for i_v, i_b in qd.ndrange(
            sap_state.fem_vertices.v.shape[0], n_envs, axes=qd.static(eoa(sap_state.fem_vertices.dp))
        ):
            if not sap_state.envs.is_linesearch_active[i_b]:
                continue
            qd.atomic_add(
                sap_state.envs.dell_dalpha[i_b],
                sap_state.fem_vertices.dp[i_v, i_b].dot(
                    sap_state.fem_vertices.v[i_v, i_b] - fem_state.vertices.vel[i_step + 1, i_v, i_b]
                ),
            )
    if qd.static(sap_config.has_rigid):
        for i_d, i_b in qd.ndrange(sap_info.n_dofs[None], n_envs, axes=qd.static(eoa(sap_state.rigid_dofs.dp))):
            if not sap_state.envs.is_linesearch_active[i_b]:
                continue
            qd.atomic_add(
                sap_state.envs.energy[i_b],
                sap_state.envs.step_size[i_b]
                * sap_state.rigid_dofs.dp[i_d, i_b]
                * (sap_state.rigid_dofs.v[i_d, i_b] - dofs_state.vel[i_d, i_b]),
            )
        for i_d, i_b in qd.ndrange(sap_info.n_dofs[None], n_envs, axes=qd.static(eoa(sap_state.rigid_dofs.dp))):
            if not sap_state.envs.is_linesearch_active[i_b]:
                continue
            qd.atomic_add(
                sap_state.envs.dell_dalpha[i_b],
                sap_state.rigid_dofs.dp[i_d, i_b] * (sap_state.rigid_dofs.v[i_d, i_b] - dofs_state.vel[i_d, i_b]),
            )
    if qd.static(sap_config.has_equality_constraints):
        constraint_handler.compute_energy_gamma_G(sap_state)
        constraint_handler.update_gradient_hessian_alpha(sap_state)
    for contact in qd.static(contact_handlers):
        contact.compute_energy_gamma_G(sap_state, fem_info, sap_info)
        contact.update_gradient_hessian_alpha(sap_state)


@qd.data_oriented
class BaseConstraintHandler(RBC):
    """
    Base class for constraint handling in SAPCoupler.
    """

    def __init__(self, stiffness: float = 1e8, beta: float = 0.1) -> None:
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
            energy=gs.qd_float,  # energy
            gamma=gs.qd_float,  # contact impulse
            G=gs.qd_float,  # Hessian matrix
            dvc=gs.qd_float,  # change in constraint velocity
        )

    @qd.func
    def compute_constraint_regularization(
        self, i_c: int, w_rms: float, time_step: float, constraint_info: qd.template()
    ):
        beta_factor = self.beta[None] ** 2 / (4.0 * qd.math.pi**2)
        dt2 = time_step**2
        k = constraint_info[i_c].k
        R = max(beta_factor * w_rms, 1.0 / (dt2 * k))
        constraint_info[i_c].R = R
        constraint_info[i_c].R_inv = 1.0 / R

    @qd.func
    def compute_constraint_gamma_G(self, i_c: int, vc: float, constraint_info: qd.template()):
        y = (constraint_info[i_c].v_hat - vc) * constraint_info[i_c].R_inv
        constraint_info[i_c].gamma = y
        constraint_info[i_c].G = constraint_info[i_c].R_inv

    @qd.func
    def compute_energy(self, sap_state: SAPState):
        """Add the energy of the constraints to the energy of their environments at the start of the line search."""
        constraints = qd.static(self.constraints)
        constraint_info = qd.static(constraints.constraint_info)
        for i_c in range(self.n_constraints[None]):
            i_b = constraints[i_c].batch_idx
            if sap_state.envs.is_linesearch_active[i_b]:
                vc = self.compute_vc(i_c, sap_state)
                self.compute_constraint_energy(i_c, vc, constraint_info)
                qd.atomic_add(sap_state.envs.prev_energy[i_b], constraint_info[i_c].energy)

    @qd.func
    def compute_constraint_energy(self, i_c: int, vc: float, constraint_info: qd.template()):
        y = (constraint_info[i_c].v_hat - vc) * constraint_info[i_c].R_inv
        constraint_info[i_c].energy = 0.5 * y**2 * constraint_info[i_c].R


@qd.data_oriented
class RigidConstraintHandler(BaseConstraintHandler):
    """
    Rigid body constraints in SAPCoupler. Currently only support joint equality constraints.
    """

    def __init__(self, simulator: "Simulator", stiffness: float = 1e8, beta: float = 0.1) -> None:
        super().__init__(stiffness, beta)
        rigid_solver = simulator.rigid_solver
        self.max_constraints = rigid_solver.n_equalities * simulator._B
        self.n_constraints = qd.field(gs.qd_int, shape=())
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
        self.constraints = self.constraint_type.field(shape=(self.max_constraints,))
        # Vectors of size one, as the rigid Jacobian solve shared with the contact handlers indexes each component
        self.Jt = qd.field(qd.types.vector(1, gs.qd_float), shape=(self.max_constraints, rigid_solver.n_dofs_))
        self.M_inv_Jt = qd.field(qd.types.vector(1, gs.qd_float), shape=(self.max_constraints, rigid_solver.n_dofs_))

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
            self.constraints[i_c].constraint_info.v_hat = -g0 * dt_inv

        func_rigid_solve_jacobian(
            self.n_constraints[None],
            self.constraints.batch_idx,
            self.Jt,
            self.M_inv_Jt,
            entities_info,
            rigid_info,
            sap_info,
            dim=1,
        )

        for i_c in range(self.n_constraints[None]):
            # Delassus operator J @ M^-1 @ J^T
            i_dof1 = self.constraints[i_c].i_dof1
            i_dof2 = self.constraints[i_c].i_dof2
            W = self.M_inv_Jt[i_c, i_dof1][0] + self.constraints[i_c].jac_dof2 * self.M_inv_Jt[i_c, i_dof2][0]
            self.compute_constraint_regularization(i_c, W, sap_info.substep_dt[None], constraint_info)

    @qd.func
    def compute_Jx(self, i_c: int, x: qd.Tensor):
        i_b = self.constraints[i_c].batch_idx
        i_dof1 = self.constraints[i_c].i_dof1
        i_dof2 = self.constraints[i_c].i_dof2
        return x[i_dof1, i_b] + self.constraints[i_c].jac_dof2 * x[i_dof2, i_b]

    @qd.func
    def add_Jt_x(self, i_c: int, x: float, y: qd.Tensor):
        i_b = self.constraints[i_c].batch_idx
        i_dof1 = self.constraints[i_c].i_dof1
        i_dof2 = self.constraints[i_c].i_dof2
        qd.atomic_add(y[i_dof1, i_b], x)
        qd.atomic_add(y[i_dof2, i_b], self.constraints[i_c].jac_dof2 * x)

    @qd.func
    def compute_vc(self, i_c: int, sap_state: SAPState):
        return self.compute_Jx(i_c, sap_state.rigid_dofs.v)

    @qd.func
    def compute_gradient_hessian_diag(self, sap_state: SAPState):
        constraints = qd.static(self.constraints)
        constraint_info = qd.static(constraints.constraint_info)
        for i_c in range(self.n_constraints[None]):
            vc = self.compute_vc(i_c, sap_state)
            self.compute_constraint_gamma_G(i_c, vc, constraint_info)
            self.add_Jt_x(i_c, -constraint_info[i_c].gamma, sap_state.rigid_dofs.gradient)
            self.add_Jt_x(i_c, constraint_info[i_c].gamma, sap_state.rigid_dofs.impulse)

    @qd.func
    def compute_Ap(self, sap_state: SAPState):
        constraints = qd.static(self.constraints)
        constraint_info = qd.static(constraints.constraint_info)
        for i_c in range(self.n_constraints[None]):
            # Jt @ G @ J @ p
            x = self.compute_Jx(i_c, sap_state.rigid_dofs.p)
            x = constraint_info[i_c].G * x
            self.add_Jt_x(i_c, x, sap_state.rigid_dofs.Ap)

    @qd.func
    def prepare_search_direction_data(self, sap_state: SAPState):
        constraints = qd.static(self.constraints)
        constraint_info = qd.static(constraints.constraint_info)
        for i_c in range(self.n_constraints[None]):
            i_b = constraints[i_c].batch_idx
            if sap_state.envs.is_linesearch_active[i_b]:
                constraint_info[i_c].dvc = self.compute_Jx(i_c, sap_state.rigid_dofs.x)

    @qd.func
    def compute_energy_gamma_G(self, sap_state: SAPState):
        constraints = qd.static(self.constraints)
        constraint_info = qd.static(constraints.constraint_info)
        for i_c in range(self.n_constraints[None]):
            vc = self.compute_vc(i_c, sap_state)
            self.compute_constraint_energy_gamma_G(i_c, vc, constraint_info)

    @qd.func
    def compute_constraint_energy_gamma_G(self, i_c: int, vc: float, constraint_info: qd.template()):
        self.compute_constraint_gamma_G(i_c, vc, constraint_info)
        constraint_info[i_c].energy = 0.5 * constraint_info[i_c].gamma ** 2 * constraint_info[i_c].R

    @qd.func
    def update_gradient_hessian_alpha(self, sap_state: SAPState):
        dvc = qd.static(self.constraints.constraint_info.dvc)
        gamma = qd.static(self.constraints.constraint_info.gamma)
        G = qd.static(self.constraints.constraint_info.G)
        for i_c in qd.ndrange(self.n_constraints[None]):
            i_b = self.constraints[i_c].batch_idx
            if sap_state.envs.is_linesearch_active[i_b]:
                qd.atomic_sub(sap_state.envs.dell_dalpha[i_b], dvc[i_c] * gamma[i_c])
                qd.atomic_add(sap_state.envs.d2ell_dalpha2[i_b], dvc[i_c] ** 2 * G[i_c])


@qd.kernel
def kernel_build_equality_constraints(
    equalities_info: EqualitiesInfo,
    joints_info: JointsInfo,
    rigid_info: RigidInfo,
    sap_info: SAPInfo,
    rigid_config: qd.template(),
    constraint_handler: qd.template(),
):
    """Build the joint equality constraints of every environment, the same in all of them."""
    constraints = qd.static(constraint_handler.constraints)
    constraint_handler.n_constraints[None] = 0
    constraint_handler.Jt.fill(0.0)
    # TODO: Maybe support different constraints for each batch in the future.
    for i_e, i_b in qd.ndrange(
        rigid_info.n_equalities[None], equalities_info.eq_type.shape[1], axes=qd.static(eoa(equalities_info.eq_type))
    ):
        if equalities_info.eq_type[i_e, i_b] == gs.EQUALITY_TYPE.JOINT:
            i_c = qd.atomic_add(constraint_handler.n_constraints[None], 1)
            constraints[i_c].batch_idx = i_b
            I_joint1 = (
                [equalities_info.eq_obj1id[i_e, i_b], i_b]
                if qd.static(rigid_config.batch_joints_info)
                else equalities_info.eq_obj1id[i_e, i_b]
            )
            I_joint2 = (
                [equalities_info.eq_obj2id[i_e, i_b], i_b]
                if qd.static(rigid_config.batch_joints_info)
                else equalities_info.eq_obj2id[i_e, i_b]
            )
            i_dof1 = joints_info.dof_start[I_joint1]
            constraints[i_c].i_eq = i_e
            constraints[i_c].i_dof1 = i_dof1
            constraints[i_c].i_dof2 = joints_info.dof_start[I_joint2]
            constraints[i_c].i_qpos1 = joints_info.q_start[I_joint1]
            constraints[i_c].i_qpos2 = joints_info.q_start[I_joint2]
            constraints[i_c].constraint_info.k = constraint_handler.stiffness[None]
            constraints[i_c].constraint_info.R_inv = sap_info.dt2[None] * constraint_handler.stiffness[None]
            constraints[i_c].constraint_info.R = 1.0 / constraints[i_c].constraint_info.R_inv
            constraints[i_c].constraint_info.v_hat = 0.0
            constraint_handler.Jt[i_c, i_dof1] = qd.Vector([1.0])


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

    def __init__(self) -> None:
        self.n_contact_pairs = qd.field(gs.qd_int, shape=())
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
            energy=gs.qd_float,  # energy
            gamma=gs.qd_vec3,  # contact impulse
            G=gs.qd_mat3,  # Hessian matrix
            dvc=gs.qd_vec3,  # velocity change at contact point, for exact line search
        )

    @qd.func
    def compute_jacobian(
        self,
        dofs_state: DofsState,
        links_state: LinksState,
        links_info: LinksInfo,
        sap_info: SAPInfo,
        rigid_config: qd.template(),
    ):
        pass

    @qd.func
    def update_gradient_hessian_alpha(self, sap_state: SAPState):
        dvc = qd.static(self.contact_pairs.contact_info.dvc)
        gamma = qd.static(self.contact_pairs.contact_info.gamma)
        G = qd.static(self.contact_pairs.contact_info.G)
        for i_p in qd.ndrange(self.n_contact_pairs[None]):
            i_b = self.contact_pairs[i_p].batch_idx
            if sap_state.envs.is_linesearch_active[i_b]:
                qd.atomic_sub(sap_state.envs.dell_dalpha[i_b], dvc[i_p].dot(gamma[i_p]))
                qd.atomic_add(sap_state.envs.d2ell_dalpha2[i_b], dvc[i_p].dot(G[i_p] @ dvc[i_p]))

    @qd.func
    def compute_delassus_world_frame(
        self,
        fem_state: FEMState,
        fem_info: FEMInfo,
        entities_info: EntitiesInfo,
        rigid_info: RigidInfo,
        sap_info: SAPInfo,
    ):
        pass

    @qd.func
    def compute_regularization(
        self,
        fem_state: FEMState,
        fem_info: FEMInfo,
        entities_info: EntitiesInfo,
        rigid_info: RigidInfo,
        sap_info: SAPInfo,
    ):
        self.compute_delassus_world_frame(fem_state, fem_info, entities_info, rigid_info, sap_info)
        for i_p in range(self.n_contact_pairs[None]):
            W = self.compute_delassus(i_p, fem_state, fem_info)
            w_rms = W.norm() / 3.0
            self.compute_contact_regularization(i_p, w_rms, self.contact_pairs.contact_info, sap_info)

    @qd.func
    def compute_energy_gamma_G(self, sap_state: SAPState, fem_info: FEMInfo, sap_info: SAPInfo):
        for i_p in range(self.n_contact_pairs[None]):
            vc = self.compute_contact_velocity(i_p, sap_state, fem_info, sap_info)
            self.compute_contact_energy_gamma_G(i_p, vc, self.contact_pairs.contact_info, sap_info)

    @qd.func
    def compute_energy(self, sap_state: SAPState, fem_info: FEMInfo, sap_info: SAPInfo):
        """Add the energy of the contacts to the energy of their environments at the start of the line search."""
        contact_info = qd.static(self.contact_pairs.contact_info)
        for i_p in range(self.n_contact_pairs[None]):
            i_b = self.contact_pairs[i_p].batch_idx
            if sap_state.envs.is_linesearch_active[i_b]:
                vc = self.compute_contact_velocity(i_p, sap_state, fem_info, sap_info)
                self.compute_contact_energy(i_p, vc, contact_info, sap_info)
                qd.atomic_add(sap_state.envs.prev_energy[i_b], contact_info[i_p].energy)

    @qd.func
    def compute_contact_gamma_G(self, i_p: int, vc: qd.types.vector(3), contact_info: qd.template(), sap_info: SAPInfo):
        y = qd.Vector([0.0, 0.0, contact_info[i_p].vn_hat]) - vc
        y[0] *= contact_info[i_p].Rt_inv
        y[1] *= contact_info[i_p].Rt_inv
        y[2] *= contact_info[i_p].Rn_inv
        yr = y[:2].norm(sap_info.EPS[None])
        yn = y[2]

        t_hat = y[:2] / yr
        contact_mode = self.compute_contact_mode(contact_info[i_p].mu, contact_info[i_p].mu_hat, yr, yn)
        contact_info[i_p].gamma = qd.Vector.zero(gs.qd_float, 3)
        contact_info[i_p].G = qd.Matrix.zero(gs.qd_float, 3, 3)
        if contact_mode == ContactMode.STICK:
            contact_info[i_p].gamma = y
            contact_info[i_p].G[0, 0] = contact_info[i_p].Rt_inv
            contact_info[i_p].G[1, 1] = contact_info[i_p].Rt_inv
            contact_info[i_p].G[2, 2] = contact_info[i_p].Rn_inv
        elif contact_mode == ContactMode.SLIDE:
            gn = (yn + contact_info[i_p].mu_hat * yr) * contact_info[i_p].mu_factor
            gt = contact_info[i_p].mu * gn * t_hat
            contact_info[i_p].gamma = qd.Vector([gt[0], gt[1], gn])
            P = t_hat.outer_product(t_hat)
            Pperp = qd.Matrix.identity(gs.qd_float, 2) - P
            dgt_dyt = contact_info[i_p].mu * (
                gn / yr * Pperp + contact_info[i_p].mu_hat * contact_info[i_p].mu_factor * P
            )
            dgt_dyn = contact_info[i_p].mu * contact_info[i_p].mu_factor * t_hat
            dgn_dyt = contact_info[i_p].mu_hat * contact_info[i_p].mu_factor * t_hat
            dgn_dyn = contact_info[i_p].mu_factor

            contact_info[i_p].G[:2, :2] = dgt_dyt * contact_info[i_p].Rt_inv
            contact_info[i_p].G[:2, 2] = dgt_dyn * contact_info[i_p].Rn_inv
            contact_info[i_p].G[2, :2] = dgn_dyt * contact_info[i_p].Rt_inv
            contact_info[i_p].G[2, 2] = dgn_dyn * contact_info[i_p].Rn_inv
        else:  # No contact
            pass

    @qd.func
    def compute_contact_energy_gamma_G(
        self, i_p: int, vc: qd.types.vector(3), contact_info: qd.template(), sap_info: SAPInfo
    ):
        self.compute_contact_gamma_G(i_p, vc, contact_info, sap_info)
        R_gamma = contact_info[i_p].gamma
        R_gamma[0] *= contact_info[i_p].Rt
        R_gamma[1] *= contact_info[i_p].Rt
        R_gamma[2] *= contact_info[i_p].Rn
        contact_info[i_p].energy = 0.5 * contact_info[i_p].gamma.dot(R_gamma)

    @qd.func
    def compute_contact_energy(self, i_p: int, vc: qd.types.vector(3), contact_info: qd.template(), sap_info: SAPInfo):
        y = qd.Vector([0.0, 0.0, contact_info[i_p].vn_hat]) - vc
        y[0] *= contact_info[i_p].Rt_inv
        y[1] *= contact_info[i_p].Rt_inv
        y[2] *= contact_info[i_p].Rn_inv
        yr = y[:2].norm(sap_info.EPS[None])
        yn = y[2]

        t_hat = y[:2] / yr
        contact_mode = self.compute_contact_mode(contact_info[i_p].mu, contact_info[i_p].mu_hat, yr, yn)
        contact_info[i_p].gamma = qd.Vector.zero(gs.qd_float, 3)
        if contact_mode == ContactMode.STICK:
            contact_info[i_p].gamma = y
        elif contact_mode == ContactMode.SLIDE:
            gn = (yn + contact_info[i_p].mu_hat * yr) * contact_info[i_p].mu_factor
            gt = contact_info[i_p].mu * gn * t_hat
            contact_info[i_p].gamma = qd.Vector([gt[0], gt[1], gn])
        else:  # No contact
            pass

        R_gamma = contact_info[i_p].gamma
        R_gamma[0] *= contact_info[i_p].Rt
        R_gamma[1] *= contact_info[i_p].Rt
        R_gamma[2] *= contact_info[i_p].Rn
        contact_info[i_p].energy = 0.5 * contact_info[i_p].gamma.dot(R_gamma)

    @qd.func
    def compute_contact_mode(self, mu: float, mu_hat: float, yr: float, yn: float):
        """
        Compute the contact mode based on the friction coefficients and the relative velocities.
        """
        result = ContactMode.NO_CONTACT
        if yr <= mu * yn:
            result = ContactMode.STICK
        elif -mu_hat * yr < yn and yn < yr / mu:
            result = ContactMode.SLIDE
        return result

    @qd.func
    def compute_contact_regularization(self, i_p: int, w_rms: float, contact_info: qd.template(), sap_info: SAPInfo):
        time_step = sap_info.substep_dt[None]
        k = contact_info[i_p].k
        Rn = max(sap_info.beta_factor[None] * w_rms, 1.0 / (time_step * k * (time_step + sap_info.taud[None])))
        Rt = sap_info.sigma[None] * w_rms
        vn_hat = -contact_info[i_p].phi0 / (time_step + sap_info.taud[None])
        contact_info[i_p].Rn = Rn
        contact_info[i_p].Rt = Rt
        contact_info[i_p].Rn_inv = 1.0 / Rn
        contact_info[i_p].Rt_inv = 1.0 / Rt
        contact_info[i_p].vn_hat = vn_hat
        contact_info[i_p].mu_hat = contact_info[i_p].mu * Rt * contact_info[i_p].Rn_inv
        contact_info[i_p].mu_factor = 1.0 / (1.0 + contact_info[i_p].mu * contact_info[i_p].mu_hat)


@qd.data_oriented
class RigidContactHandler(BaseContactHandler):
    # FIXME This function is similar to 'add_collision_constraints' in rigid/constraint/solver.py.
    # Consider refactoring, using better naming, and removing while.
    @qd.func
    def compute_jacobian(
        self,
        dofs_state: DofsState,
        links_state: LinksState,
        links_info: LinksInfo,
        sap_info: SAPInfo,
        rigid_config: qd.template(),
    ):
        for i_p in range(self.n_contact_pairs[None]):
            link = self.contact_pairs[i_p].link_idx
            i_b = self.contact_pairs[i_p].batch_idx
            for i_d in range(sap_info.n_dofs[None]):
                self.Jt[i_p, i_d] = qd.Vector.zero(gs.qd_float, 3)
            while link > -1:
                link_maybe_batch = [link, i_b] if qd.static(rigid_config.batch_links_info) else link
                # reverse order to make sure dofs in each row of self.jac_dofs_idx is strictly descending
                for i_d_ in range(links_info.n_dofs[link_maybe_batch]):
                    i_d = links_info.dof_end[link_maybe_batch] - 1 - i_d_

                    cdof_ang = dofs_state.cdof_ang[i_d, i_b]
                    cdof_vel = dofs_state.cdof_vel[i_d, i_b]

                    t_quat = gu.qd_identity_quat()
                    t_pos = self.contact_pairs[i_p].contact_pos - links_state.root_COM[link, i_b]
                    _, vel = gu.qd_transform_motion_by_trans_quat(cdof_ang, cdof_vel, t_pos, t_quat)

                    diff = vel
                    jac = diff
                    self.Jt[i_p, i_d] = self.Jt[i_p, i_d] + jac
                link = links_info.parent_idx[link_maybe_batch]

    @qd.func
    def compute_gradient_hessian_diag(
        self,
        sap_state: SAPState,
        fem_info: FEMInfo,
        vertex_constraints: FEMVertexConstraints,
        sap_info: SAPInfo,
        fem_config: qd.template(),
    ):
        contact_info = qd.static(self.contact_pairs.contact_info)
        for i_p in range(self.n_contact_pairs[None]):
            vc = self.compute_contact_velocity(i_p, sap_state, fem_info, sap_info)
            self.compute_contact_gamma_G(i_p, vc, contact_info, sap_info)
            self.add_Jt_x(i_p, -contact_info[i_p].gamma, sap_state.rigid_dofs.gradient, sap_info)
            self.add_Jt_x(i_p, contact_info[i_p].gamma, sap_state.rigid_dofs.impulse, sap_info)

    @qd.func
    def compute_pcg_matrix_vector_product(
        self,
        sap_state: SAPState,
        fem_info: FEMInfo,
        vertex_constraints: FEMVertexConstraints,
        sap_info: SAPInfo,
        fem_config: qd.template(),
    ):
        contact_info = qd.static(self.contact_pairs.contact_info)
        for i_p in range(self.n_contact_pairs[None]):
            # Jt @ G @ J @ p
            Jp = self.compute_Jx(i_p, sap_state.rigid_dofs.p, sap_info)
            GJp = contact_info[i_p].G @ Jp
            self.add_Jt_x(i_p, GJp, sap_state.rigid_dofs.Ap, sap_info)

    @qd.func
    def compute_contact_velocity(self, i_p: int, sap_state: SAPState, fem_info: FEMInfo, sap_info: SAPInfo):
        """
        Compute the contact velocity in the contact frame.
        """
        return self.compute_Jx(i_p, sap_state.rigid_dofs.v, sap_info)

    @qd.func
    def prepare_search_direction_data(self, sap_state: SAPState, fem_info: FEMInfo, sap_info: SAPInfo):
        contact_info = qd.static(self.contact_pairs.contact_info)
        for i_p in qd.ndrange(self.n_contact_pairs[None]):
            i_b = self.contact_pairs[i_p].batch_idx
            if sap_state.envs.is_linesearch_active[i_b]:
                contact_info[i_p].dvc = self.compute_Jx(i_p, sap_state.rigid_dofs.x, sap_info)

    @qd.func
    def compute_delassus_world_frame(
        self,
        fem_state: FEMState,
        fem_info: FEMInfo,
        entities_info: EntitiesInfo,
        rigid_info: RigidInfo,
        sap_info: SAPInfo,
    ):
        func_rigid_solve_jacobian(
            self.n_contact_pairs[None],
            self.contact_pairs.batch_idx,
            self.Jt,
            self.M_inv_Jt,
            entities_info,
            rigid_info,
            sap_info,
            dim=3,
        )
        for i_p in range(self.n_contact_pairs[None]):
            W = qd.Matrix.zero(gs.qd_float, 3, 3)
            for i_d in range(sap_info.n_dofs[None]):
                W = W + self.M_inv_Jt[i_p, i_d].outer_product(self.Jt[i_p, i_d])
            self.W[i_p] = W

    @qd.func
    def compute_delassus(self, i_p: int, fem_state: FEMState, fem_info: FEMInfo):
        return self.W[i_p]

    @qd.func
    def compute_Jx(self, i_p: int, x: qd.Tensor, sap_info: SAPInfo):
        pairs = qd.static(self.contact_pairs)
        i_b = pairs[i_p].batch_idx
        Jx = qd.Vector.zero(gs.qd_float, 3)
        for i in range(sap_info.n_dofs[None]):
            Jx = Jx + self.Jt[i_p, i] * x[i, i_b]
        return Jx

    @qd.func
    def add_Jt_x(self, i_p: int, x: qd.types.vector(3), y: qd.Tensor, sap_info: SAPInfo):
        pairs = qd.static(self.contact_pairs)
        i_b = pairs[i_p].batch_idx
        for i in range(sap_info.n_dofs[None]):
            qd.atomic_add(y[i, i_b], self.Jt[i_p, i].dot(x))


@qd.data_oriented
class RigidRigidContactHandler(RigidContactHandler):
    @qd.func
    def compute_jacobian(
        self,
        dofs_state: DofsState,
        links_state: LinksState,
        links_info: LinksInfo,
        sap_info: SAPInfo,
        rigid_config: qd.template(),
    ):
        pairs = qd.static(self.contact_pairs)
        for i_p in range(self.n_contact_pairs[None]):
            i_b = pairs[i_p].batch_idx
            for i_d in range(sap_info.n_dofs[None]):
                self.Jt[i_p, i_d] = qd.Vector.zero(gs.qd_float, 3)
            link = pairs[i_p].link_idx0
            while link > -1:
                link_maybe_batch = [link, i_b] if qd.static(rigid_config.batch_links_info) else link
                # reverse order to make sure dofs in each row of self.jac_dofs_idx is strictly descending
                for i_d_ in range(links_info.n_dofs[link_maybe_batch]):
                    i_d = links_info.dof_end[link_maybe_batch] - 1 - i_d_

                    cdof_ang = dofs_state.cdof_ang[i_d, i_b]
                    cdof_vel = dofs_state.cdof_vel[i_d, i_b]

                    t_quat = gu.qd_identity_quat()
                    t_pos = pairs[i_p].contact_pos - links_state.root_COM[link, i_b]
                    _, vel = gu.qd_transform_motion_by_trans_quat(cdof_ang, cdof_vel, t_pos, t_quat)

                    self.Jt[i_p, i_d] = self.Jt[i_p, i_d] + vel
                link = links_info.parent_idx[link_maybe_batch]
            link = pairs[i_p].link_idx1
            while link > -1:
                link_maybe_batch = [link, i_b] if qd.static(rigid_config.batch_links_info) else link
                # reverse order to make sure dofs in each row of self.jac_dofs_idx is strictly descending
                for i_d_ in range(links_info.n_dofs[link_maybe_batch]):
                    i_d = links_info.dof_end[link_maybe_batch] - 1 - i_d_

                    cdof_ang = dofs_state.cdof_ang[i_d, i_b]
                    cdof_vel = dofs_state.cdof_vel[i_d, i_b]

                    t_quat = gu.qd_identity_quat()
                    t_pos = pairs[i_p].contact_pos - links_state.root_COM[link, i_b]
                    _, vel = gu.qd_transform_motion_by_trans_quat(cdof_ang, cdof_vel, t_pos, t_quat)

                    self.Jt[i_p, i_d] = self.Jt[i_p, i_d] - vel
                link = links_info.parent_idx[link_maybe_batch]

    @qd.func
    def compute_delassus(self, i_p: int, fem_state: FEMState, fem_info: FEMInfo):
        pairs = qd.static(self.contact_pairs)
        world = qd.Matrix.cols([pairs[i_p].tangent0, pairs[i_p].tangent1, pairs[i_p].normal])
        return world.transpose() @ self.W[i_p] @ world

    @qd.func
    def compute_Jx(self, i_p: int, x: qd.Tensor, sap_info: SAPInfo):
        pairs = qd.static(self.contact_pairs)
        i_b = pairs[i_p].batch_idx
        Jx = qd.Vector.zero(gs.qd_float, 3)
        for i in range(sap_info.n_dofs[None]):
            Jx = Jx + self.Jt[i_p, i] * x[i, i_b]
        Jx = qd.Vector([Jx.dot(pairs[i_p].tangent0), Jx.dot(pairs[i_p].tangent1), Jx.dot(pairs[i_p].normal)])
        return Jx

    @qd.func
    def add_Jt_x(self, i_p: int, x: qd.types.vector(3), y: qd.Tensor, sap_info: SAPInfo):
        pairs = qd.static(self.contact_pairs)
        i_b = pairs[i_p].batch_idx
        world = qd.Matrix.cols([pairs[i_p].tangent0, pairs[i_p].tangent1, pairs[i_p].normal])
        x_ = world @ x
        for i in range(sap_info.n_dofs[None]):
            qd.atomic_add(y[i, i_b], self.Jt[i_p, i].dot(x_))


@qd.data_oriented
class FEMContactHandler(BaseContactHandler):
    @qd.func
    def compute_gradient_hessian_diag(
        self,
        sap_state: SAPState,
        fem_info: FEMInfo,
        vertex_constraints: FEMVertexConstraints,
        sap_info: SAPInfo,
        fem_config: qd.template(),
    ):
        contact_info = qd.static(self.contact_pairs.contact_info)
        for i_p in range(self.n_contact_pairs[None]):
            vc = self.compute_Jx(i_p, sap_state.fem_vertices.v, fem_info)
            self.compute_contact_gamma_G(i_p, vc, contact_info, sap_info)
            self.add_Jt_x(
                i_p,
                -contact_info[i_p].gamma,
                sap_state.fem_vertices.gradient,
                fem_info,
                vertex_constraints,
                fem_config,
            )
            self.add_Jt_x(
                i_p,
                contact_info[i_p].gamma,
                sap_state.fem_vertices.impulse,
                fem_info,
                vertex_constraints,
                fem_config,
            )
            self.add_Jt_A_J_diag3x3(
                i_p,
                contact_info[i_p].G,
                sap_state.fem_vertices.diag3x3,
                fem_info,
                vertex_constraints,
                fem_config,
            )

    @qd.func
    def prepare_search_direction_data(self, sap_state: SAPState, fem_info: FEMInfo, sap_info: SAPInfo):
        contact_info = qd.static(self.contact_pairs.contact_info)
        for i_p in qd.ndrange(self.n_contact_pairs[None]):
            i_b = self.contact_pairs[i_p].batch_idx
            if sap_state.envs.is_linesearch_active[i_b]:
                contact_info[i_p].dvc = self.compute_Jx(i_p, sap_state.fem_vertices.x, fem_info)

    @qd.func
    def compute_pcg_matrix_vector_product(
        self,
        sap_state: SAPState,
        fem_info: FEMInfo,
        vertex_constraints: FEMVertexConstraints,
        sap_info: SAPInfo,
        fem_config: qd.template(),
    ):
        contact_info = qd.static(self.contact_pairs.contact_info)
        for i_p in range(self.n_contact_pairs[None]):
            # Jt @ G @ J @ p
            x = self.compute_Jx(i_p, sap_state.fem_vertices.p, fem_info)
            x = contact_info[i_p].G @ x
            self.add_Jt_x(i_p, x, sap_state.fem_vertices.Ap, fem_info, vertex_constraints, fem_config)

    @qd.func
    def compute_contact_velocity(self, i_p: int, sap_state: SAPState, fem_info: FEMInfo, sap_info: SAPInfo):
        """
        Compute the contact velocity in the contact frame.
        """
        return self.compute_Jx(i_p, sap_state.fem_vertices.v, fem_info)


@qd.data_oriented
class RigidFEMContactHandler(RigidContactHandler):
    @qd.func
    def compute_gradient_hessian_diag(
        self,
        sap_state: SAPState,
        fem_info: FEMInfo,
        vertex_constraints: FEMVertexConstraints,
        sap_info: SAPInfo,
        fem_config: qd.template(),
    ):
        contact_info = qd.static(self.contact_pairs.contact_info)
        for i_p in range(self.n_contact_pairs[None]):
            vc = self.compute_Jx(i_p, sap_state.fem_vertices.v, sap_state.rigid_dofs.v, fem_info, sap_info)
            self.compute_contact_gamma_G(i_p, vc, contact_info, sap_info)
            self.add_Jt_x(
                i_p,
                -contact_info[i_p].gamma,
                sap_state.fem_vertices.gradient,
                sap_state.rigid_dofs.gradient,
                fem_info,
                sap_info,
            )
            self.add_Jt_x(
                i_p,
                contact_info[i_p].gamma,
                sap_state.fem_vertices.impulse,
                sap_state.rigid_dofs.impulse,
                fem_info,
                sap_info,
            )
            self.add_Jt_A_J_diag3x3(i_p, contact_info[i_p].G, sap_state.fem_vertices.diag3x3, fem_info)

    @qd.func
    def prepare_search_direction_data(self, sap_state: SAPState, fem_info: FEMInfo, sap_info: SAPInfo):
        contact_info = qd.static(self.contact_pairs.contact_info)
        for i_p in qd.ndrange(self.n_contact_pairs[None]):
            i_b = self.contact_pairs[i_p].batch_idx
            if sap_state.envs.is_linesearch_active[i_b]:
                contact_info[i_p].dvc = self.compute_Jx(
                    i_p, sap_state.fem_vertices.x, sap_state.rigid_dofs.x, fem_info, sap_info
                )

    @qd.func
    def compute_pcg_matrix_vector_product(
        self,
        sap_state: SAPState,
        fem_info: FEMInfo,
        vertex_constraints: FEMVertexConstraints,
        sap_info: SAPInfo,
        fem_config: qd.template(),
    ):
        contact_info = qd.static(self.contact_pairs.contact_info)
        for i_p in range(self.n_contact_pairs[None]):
            # Jt @ G @ J @ p
            x = self.compute_Jx(i_p, sap_state.fem_vertices.p, sap_state.rigid_dofs.p, fem_info, sap_info)
            x = contact_info[i_p].G @ x
            self.add_Jt_x(i_p, x, sap_state.fem_vertices.Ap, sap_state.rigid_dofs.Ap, fem_info, sap_info)

    @qd.func
    def compute_contact_velocity(self, i_p: int, sap_state: SAPState, fem_info: FEMInfo, sap_info: SAPInfo):
        """
        Compute the contact velocity in the contact frame.
        """
        return self.compute_Jx(i_p, sap_state.fem_vertices.v, sap_state.rigid_dofs.v, fem_info, sap_info)


@qd.func
def accumulate_area_centroid(
    polygon_vertices, i, total_area: qd.template(), total_area_weighted_centroid: qd.template()
):
    e1 = polygon_vertices[:, i - 1] - polygon_vertices[:, 0]
    e2 = polygon_vertices[:, i] - polygon_vertices[:, 0]
    area = 0.5 * e1.cross(e2).norm()
    total_area += area
    total_area_weighted_centroid += (
        area * (polygon_vertices[:, 0] + polygon_vertices[:, i - 1] + polygon_vertices[:, i]) / 3.0
    )


@qd.data_oriented
class FEMFloorTetContactHandler(FEMContactHandler):
    """
    Class for handling contact between a tetrahedral mesh and a floor in a simulation using hydroelastic model.

    This class extends the BaseContact class and provides methods for detecting contact
    between the tetrahedral elements and the floor, computing contact pairs, and managing
    contact-related computations.
    """

    def __init__(self, simulator: "Simulator") -> None:
        super().__init__()
        fem_solver = simulator.fem_solver
        self.name = "FEMFloorTetContactHandler"
        self.contact_candidate_type = qd.types.struct(
            batch_idx=gs.qd_int,  # batch index
            geom_idx=gs.qd_int,  # index of the FEM element
            intersection_code=gs.qd_int,  # intersection code for the element
            distance=gs.qd_vec4,  # distance vector for the element
        )
        self.n_contact_candidates = qd.field(gs.qd_int, shape=())
        self.max_contact_candidates = fem_solver.n_surface_elements * fem_solver._B
        self.contact_candidates = self.contact_candidate_type.field(shape=(self.max_contact_candidates,))

        self.contact_pair_type = qd.types.struct(
            batch_idx=gs.qd_int,  # batch index
            geom_idx=gs.qd_int,  # index of the FEM element
            barycentric=gs.qd_vec4,  # barycentric coordinates of the contact point
            contact_pos=gs.qd_vec3,  # contact position
            contact_info=self.contact_info_type,
        )
        self.max_contact_pairs = fem_solver.n_surface_elements * fem_solver._B
        self.contact_pairs = self.contact_pair_type.field(shape=(self.max_contact_pairs,))

    @qd.func
    def detection(
        self,
        f: int,
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
    ):
        has_overflow = False
        # Compute contact pairs
        self.n_contact_candidates[None] = 0
        # TODO Check surface element only instead of all elements
        for i_e, i_b in qd.ndrange(
            fem_info.elements.V.shape[0], sap_state.envs.is_active.shape[0], axes=qd.static(eoa(geoms_state.pos))
        ):
            i_g = sap_info.ground_geom_idx[None]
            ground_pos = geoms_state.pos[i_g, i_b]
            intersection_code = qd.int32(0)
            distance = qd.Vector.zero(gs.qd_float, 4)
            for i in qd.static(range(4)):
                i_v = fem_info.elements.el2v[i_e][i]
                pos_v = fem_state.vertices.pos[f, i_v, i_b]
                distance[i] = pos_v.z - ground_pos.z
                if distance[i] > 0.0:
                    intersection_code |= 1 << i

            # check if the element intersect with the floor
            if intersection_code != 0 and intersection_code != 15:
                i_c = qd.atomic_add(self.n_contact_candidates[None], 1)
                if i_c < self.max_contact_candidates:
                    self.contact_candidates[i_c].batch_idx = i_b
                    self.contact_candidates[i_c].geom_idx = i_e
                    self.contact_candidates[i_c].intersection_code = intersection_code
                    self.contact_candidates[i_c].distance = distance
                else:
                    has_overflow = True

        contact_info = qd.static(self.contact_pairs.contact_info)
        self.n_contact_pairs[None] = 0
        # Compute pair from candidates
        result_count = qd.min(self.n_contact_candidates[None], self.max_contact_candidates)
        for i_c in range(result_count):
            candidate = self.contact_candidates[i_c]
            i_b = candidate.batch_idx
            i_e = candidate.geom_idx
            intersection_code = candidate.intersection_code
            intersected_edges = sap_info.marching_tets_edge_table[intersection_code]

            tet_vertices = qd.Matrix.zero(gs.qd_float, 3, 4)  # 4 vertices
            tet_pressures = qd.Vector.zero(gs.qd_float, 4)  # pressures at the vertices
            for i in qd.static(range(4)):
                i_v = fem_info.elements.el2v[i_e][i]
                tet_vertices[:, i] = fem_state.vertices.pos[f, i_v, i_b]
                tet_pressures[i] = sap_info.fem_pressure[i_v]

            polygon_vertices = qd.Matrix.zero(gs.qd_float, 3, 4)  # 3 or 4 vertices
            total_area = sap_info.EPS[None]  # avoid division by zero
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

            centroid = total_area_weighted_centroid / total_area

            # Compute barycentric coordinates
            barycentric = tet_barycentric(centroid, tet_vertices)
            pressure = barycentric.dot(tet_pressures)

            deformable_g = sap_info.hydroelastic_stiffness[None]
            rigid_g = sap_state.fem_pressure_gradient[i_e, i_b].z
            # TODO A better way to handle corner cases where pressure and pressure gradient are ill defined
            if total_area < sap_info.contact_eps[None] or rigid_g < sap_info.contact_eps[None]:
                continue
            g = 1.0 / (1.0 / deformable_g + 1.0 / rigid_g)  # harmonic average
            rigid_k = total_area * g
            rigid_phi0 = -pressure / g
            if rigid_k < sap_info.contact_eps[None] or rigid_phi0 > sap_info.contact_eps[None]:
                continue
            i_p = qd.atomic_add(self.n_contact_pairs[None], 1)
            if i_p < self.max_contact_pairs:
                self.contact_pairs[i_p].batch_idx = i_b
                self.contact_pairs[i_p].geom_idx = i_e
                self.contact_pairs[i_p].barycentric = barycentric
                contact_info[i_p].k = rigid_k
                contact_info[i_p].phi0 = rigid_phi0
                contact_info[i_p].mu = fem_info.elements.friction_mu[i_e]
            else:
                has_overflow = True

        return has_overflow

    @qd.func
    def compute_Jx(self, i_p: int, x: qd.Tensor, fem_info: FEMInfo):
        """
        Compute the contact Jacobian J times a vector x.
        """
        i_b = self.contact_pairs[i_p].batch_idx
        i_g = self.contact_pairs[i_p].geom_idx
        Jx = qd.Vector.zero(gs.qd_float, 3)
        for i in qd.static(range(4)):
            i_v = fem_info.elements.el2v[i_g][i]
            Jx += self.contact_pairs[i_p].barycentric[i] * x[i_v, i_b]
        return Jx

    @qd.func
    def add_Jt_x(
        self,
        i_p: int,
        x: qd.types.vector(3),
        y: qd.Tensor,
        fem_info: FEMInfo,
        vertex_constraints: FEMVertexConstraints,
        fem_config: qd.template(),
    ):
        i_b = self.contact_pairs[i_p].batch_idx
        i_g = self.contact_pairs[i_p].geom_idx
        for i in qd.static(range(4)):
            i_v = fem_info.elements.el2v[i_g][i]
            if qd.static(fem_config.enable_vertex_constraints):
                if not vertex_constraints.is_constrained[i_v, i_b]:
                    qd.atomic_add(y[i_v, i_b], self.contact_pairs[i_p].barycentric[i] * x)
            else:
                qd.atomic_add(y[i_v, i_b], self.contact_pairs[i_p].barycentric[i] * x)

    @qd.func
    def add_Jt_A_J_diag3x3(
        self,
        i_p: int,
        A: qd.types.matrix(3, 3),
        y: qd.Tensor,
        fem_info: FEMInfo,
        vertex_constraints: FEMVertexConstraints,
        fem_config: qd.template(),
    ):
        i_b = self.contact_pairs[i_p].batch_idx
        i_g = self.contact_pairs[i_p].geom_idx
        for i in qd.static(range(4)):
            i_v = fem_info.elements.el2v[i_g][i]
            if qd.static(fem_config.enable_vertex_constraints):
                if not vertex_constraints.is_constrained[i_v, i_b]:
                    qd.atomic_add(y[i_v, i_b], self.contact_pairs[i_p].barycentric[i] ** 2 * A)
            else:
                qd.atomic_add(y[i_v, i_b], self.contact_pairs[i_p].barycentric[i] ** 2 * A)

    @qd.func
    def compute_delassus(self, i_p: int, fem_state: FEMState, fem_info: FEMInfo):
        dt2_inv = fem_info.dt2_inv[None]
        i_b = self.contact_pairs[i_p].batch_idx
        i_g = self.contact_pairs[i_p].geom_idx
        W = qd.Matrix.zero(gs.qd_float, 3, 3)
        # W = sum (JA^-1J^T)
        # With floor, J is Identity times the barycentric coordinates
        for i in qd.static(range(4)):
            i_v = fem_info.elements.el2v[i_g][i]
            W += self.contact_pairs[i_p].barycentric[i] ** 2 * dt2_inv * fem_state.vertices_solve.prec[i_v, i_b]
        return W


@qd.data_oriented
class FEMSelfTetContactHandler(FEMContactHandler):
    """
    Class for handling self-contact between tetrahedral elements in a simulation using hydroelastic model.

    This class extends the FEMContact class and provides methods for detecting self-contact
    between tetrahedral elements, computing contact pairs, and managing contact-related computations.
    """

    def __init__(self, simulator: "Simulator") -> None:
        super().__init__()
        fem_solver = simulator.fem_solver
        self.name = "FEMSelfTetContactHandler"
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
        self.max_contact_candidates = fem_solver.n_surface_elements * fem_solver._B * 8
        self.contact_candidates = self.contact_candidate_type.field(shape=(self.max_contact_candidates,))

        self.contact_pair_type = qd.types.struct(
            batch_idx=gs.qd_int,  # batch index
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
        self.max_contact_pairs = fem_solver.n_surface_elements * fem_solver._B
        self.contact_pairs = self.contact_pair_type.field(shape=(self.max_contact_pairs,))

    @qd.func
    def compute_candidates(
        self,
        f: int,
        fem_state: FEMState,
        sap_state: SAPState,
        query_results: BVHQueryResults,
        fem_info: FEMInfo,
        sap_info: SAPInfo,
    ):
        has_overflow = False
        self.n_contact_candidates[None] = 0
        result_count = qd.min(query_results.count[0], query_results.triplets.shape[0])
        for i_r in range(result_count):
            i_b, i_sa, i_sq = query_results.triplets[i_r]
            i_a = fem_info.surface_elements[i_sa]
            i_q = fem_info.surface_elements[i_sq]
            i_v0 = fem_info.elements.el2v[i_a][0]
            i_v1 = fem_info.elements.el2v[i_q][0]
            x0 = fem_state.vertices.pos[f, i_v0, i_b]
            x1 = fem_state.vertices.pos[f, i_v1, i_b]
            p0 = sap_info.fem_pressure[i_v0]
            p1 = sap_info.fem_pressure[i_v1]
            g0 = sap_state.fem_pressure_gradient[i_a, i_b]
            g1 = sap_state.fem_pressure_gradient[i_q, i_b]
            g0_norm = g0.norm()
            g1_norm = g1.norm()
            if g0_norm < sap_info.EPS[None] or g1_norm < sap_info.EPS[None]:
                continue
            # Calculate the isosurface, i.e. equal pressure plane defined by x and normal
            # Solve for p0 + g0.dot(x - x0) = p1 + g1.dot(x - x1)
            normal = g0 - g1
            magnitude = normal.norm()
            if magnitude < sap_info.EPS[None]:
                continue
            normal /= magnitude
            b = p1 - p0 - g1.dot(x1) + g0.dot(x0)
            x = b / magnitude * normal
            # Check that the normal is pointing along g0 and against g1, some allowance as used in Drake
            threshold = sap_info.cos_angle_threshold[None]
            if normal.dot(g0) < threshold * g0_norm or normal.dot(g1) > -threshold * g1_norm:
                continue
            intersection_code0 = qd.int32(0)
            distance0 = qd.Vector.zero(gs.qd_float, 4)
            intersection_code1 = qd.int32(0)
            distance1 = qd.Vector.zero(gs.qd_float, 4)
            for i in qd.static(range(4)):
                i_v = fem_info.elements.el2v[i_a][i]
                pos_v = fem_state.vertices.pos[f, i_v, i_b]
                distance0[i] = (pos_v - x).dot(normal)  # signed distance
                if distance0[i] > 0.0:
                    intersection_code0 |= 1 << i
            for i in qd.static(range(4)):
                i_v = fem_info.elements.el2v[i_q][i]
                pos_v = fem_state.vertices.pos[f, i_v, i_b]
                distance1[i] = (pos_v - x).dot(normal)
                if distance1[i] > 0.0:
                    intersection_code1 |= 1 << i
            # Fast check for whether both tets intersect with the plane
            if (
                intersection_code0 == 0
                or intersection_code1 == 0
                or intersection_code0 == 15
                or intersection_code1 == 15
            ):
                continue
            i_c = qd.atomic_add(self.n_contact_candidates[None], 1)
            if i_c < self.max_contact_candidates:
                self.contact_candidates[i_c].batch_idx = i_b
                self.contact_candidates[i_c].normal = normal
                self.contact_candidates[i_c].x = x
                self.contact_candidates[i_c].geom_idx0 = i_a
                self.contact_candidates[i_c].intersection_code0 = intersection_code0
                self.contact_candidates[i_c].distance0 = distance0
                self.contact_candidates[i_c].geom_idx1 = i_q
            else:
                has_overflow = True
        return has_overflow

    @qd.func
    def compute_pairs(
        self, i_step: int, fem_state: FEMState, sap_state: SAPState, fem_info: FEMInfo, sap_info: SAPInfo
    ):
        """
        Computes the FEM self contact pairs and their properties.

        Intersection code reference:
        https://github.com/RobotLocomotion/drake/blob/8c3a249184ed09f0faab3c678536d66d732809ce/geometry/proximity/field_intersection.cc#L87
        """
        has_overflow = False
        contact_info = qd.static(self.contact_pairs.contact_info)
        normal_signs = qd.Vector([1.0, -1.0, 1.0, -1.0], dt=gs.qd_float)  # make normal point outward
        self.n_contact_pairs[None] = 0
        result_count = qd.min(self.n_contact_candidates[None], self.max_contact_candidates)
        for i_c in range(result_count):
            i_b = self.contact_candidates[i_c].batch_idx
            i_e0 = self.contact_candidates[i_c].geom_idx0
            i_e1 = self.contact_candidates[i_c].geom_idx1
            intersection_code0 = self.contact_candidates[i_c].intersection_code0
            distance0 = self.contact_candidates[i_c].distance0
            intersected_edges0 = sap_info.marching_tets_edge_table[intersection_code0]

            tet_vertices0 = qd.Matrix.zero(gs.qd_float, 3, 4)  # 4 vertices of tet 0
            tet_pressures0 = qd.Vector.zero(gs.qd_float, 4)  # pressures at the vertices of tet 0
            tet_vertices1 = qd.Matrix.zero(gs.qd_float, 3, 4)  # 4 vertices of tet 1
            for i in qd.static(range(4)):
                i_v = fem_info.elements.el2v[i_e0][i]
                tet_vertices0[:, i] = fem_state.vertices.pos[i_step, i_v, i_b]
                tet_pressures0[i] = sap_info.fem_pressure[i_v]
            for i in qd.static(range(4)):
                i_v = fem_info.elements.el2v[i_e1][i]
                tet_vertices1[:, i] = fem_state.vertices.pos[i_step, i_v, i_b]

            polygon_vertices = qd.Matrix.zero(gs.qd_float, 3, 8)  # maximum 8 vertices
            polygon_n_vertices = gs.qd_int(0)
            clipped_vertices = qd.Matrix.zero(gs.qd_float, 3, 8)  # maximum 8 vertices
            clipped_n_vertices = gs.qd_int(0)
            for i in range(4):
                if intersected_edges0[i] >= 0:
                    edge = sap_info.tet_edges[intersected_edges0[i]]
                    pos_v0 = tet_vertices0[:, edge[0]]
                    pos_v1 = tet_vertices0[:, edge[1]]
                    d_v0 = distance0[edge[0]]
                    d_v1 = distance0[edge[1]]
                    t = d_v0 / (d_v0 - d_v1)
                    polygon_vertices[:, polygon_n_vertices] = pos_v0 + t * (pos_v1 - pos_v0)
                    polygon_n_vertices += 1
            # Intersects the polygon with the four halfspaces of the four triangles
            # of the tetrahedral element1.
            for face in range(4):
                clipped_n_vertices = 0
                x = tet_vertices1[:, (face + 1) % 4]
                normal = (tet_vertices1[:, (face + 2) % 4] - x).cross(
                    tet_vertices1[:, (face + 3) % 4] - x
                ) * normal_signs[face]
                normal /= normal.norm()

                distances = qd.Vector.zero(gs.qd_float, 8)
                for i in range(polygon_n_vertices):
                    distances[i] = (polygon_vertices[:, i] - x).dot(normal)

                for i in range(polygon_n_vertices):
                    j = (i + 1) % polygon_n_vertices
                    if distances[i] <= 0.0:
                        clipped_vertices[:, clipped_n_vertices] = polygon_vertices[:, i]
                        clipped_n_vertices += 1
                        if distances[j] > 0.0:
                            wa = distances[j] / (distances[j] - distances[i])
                            wb = 1.0 - wa
                            clipped_vertices[:, clipped_n_vertices] = (
                                wa * polygon_vertices[:, i] + wb * polygon_vertices[:, j]
                            )
                            clipped_n_vertices += 1
                    elif distances[j] <= 0.0:
                        wa = distances[j] / (distances[j] - distances[i])
                        wb = 1.0 - wa
                        clipped_vertices[:, clipped_n_vertices] = (
                            wa * polygon_vertices[:, i] + wb * polygon_vertices[:, j]
                        )
                        clipped_n_vertices += 1
                polygon_n_vertices = clipped_n_vertices
                polygon_vertices = clipped_vertices

                if polygon_n_vertices < 3:
                    # If the polygon has less than 3 vertices, it is not a valid contact
                    break

            if polygon_n_vertices < 3:
                continue

            # compute centroid and area of the polygon
            total_area = 0.0
            total_area_weighted_centroid = qd.Vector.zero(gs.qd_float, 3)
            for i in range(2, polygon_n_vertices):
                accumulate_area_centroid(polygon_vertices, i, total_area, total_area_weighted_centroid)

            if total_area < sap_info.contact_eps[None]:
                continue
            centroid = total_area_weighted_centroid / total_area
            barycentric0 = tet_barycentric(centroid, tet_vertices0)
            barycentric1 = tet_barycentric(centroid, tet_vertices1)
            tangent0 = polygon_vertices[:, 0] - centroid
            tangent0 /= tangent0.norm()
            tangent1 = self.contact_candidates[i_c].normal.cross(tangent0)

            pressure = barycentric0.dot(tet_pressures0)
            g0 = sap_state.fem_pressure_gradient[i_e0, i_b].dot(self.contact_candidates[i_c].normal)
            g1 = -sap_state.fem_pressure_gradient[i_e1, i_b].dot(self.contact_candidates[i_c].normal)
            # The stiffness of a contact is its area times the pressure gradients of both sides along its normal,
            # harmonically averaged, as for the other hydroelastic contacts. A side whose pressure does not increase
            # into its body along the normal has no stiffness to give.
            if g0 < sap_info.contact_eps[None] or g1 < sap_info.contact_eps[None]:
                continue
            g = 1.0 / (1.0 / g0 + 1.0 / g1)  # harmonic average
            # FIXME This is an approximated value, different from Drake, which actually calculates the distance
            deformable_phi0 = -pressure / g

            if deformable_phi0 > sap_info.EPS[None]:
                continue

            i_p = qd.atomic_add(self.n_contact_pairs[None], 1)
            if i_p < self.max_contact_pairs:
                self.contact_pairs[i_p].batch_idx = i_b
                self.contact_pairs[i_p].normal = self.contact_candidates[i_c].normal
                self.contact_pairs[i_p].tangent0 = tangent0
                self.contact_pairs[i_p].tangent1 = tangent1
                self.contact_pairs[i_p].geom_idx0 = i_e0
                self.contact_pairs[i_p].geom_idx1 = i_e1
                self.contact_pairs[i_p].barycentric0 = barycentric0
                self.contact_pairs[i_p].barycentric1 = barycentric1

                contact_info[i_p].k = total_area * g
                contact_info[i_p].phi0 = deformable_phi0
                contact_info[i_p].mu = qd.sqrt(
                    fem_info.elements.friction_mu[i_e0] * fem_info.elements.friction_mu[i_e1]
                )
            else:
                has_overflow = True
        return has_overflow

    @qd.func
    def detection(
        self,
        f: int,
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
    ):
        has_overflow = False
        contact_queries_state.fem_self.results.count[0] = 0
        func_bvh_query_leaves(
            contact_queries_state.fem_self,
            func_filter_fem_surface_tets,
            filter_ctx=(fem_info.surface_elements, fem_info.elements.el2v),
        )
        has_overflow |= (
            contact_queries_state.fem_self.results.count[0] > contact_queries_state.fem_self.results.triplets.shape[0]
        )
        has_overflow |= self.compute_candidates(
            f, fem_state, sap_state, contact_queries_state.fem_self.results, fem_info, sap_info
        )
        has_overflow |= self.compute_pairs(f, fem_state, sap_state, fem_info, sap_info)
        return has_overflow

    @qd.func
    def compute_Jx(self, i_p: int, x: qd.Tensor, fem_info: FEMInfo):
        """
        Compute the contact Jacobian J times a vector x.
        """
        i_b = self.contact_pairs[i_p].batch_idx
        i_g0 = self.contact_pairs[i_p].geom_idx0
        i_g1 = self.contact_pairs[i_p].geom_idx1
        Jx = qd.Vector.zero(gs.qd_float, 3)
        for i in qd.static(range(4)):
            i_v = fem_info.elements.el2v[i_g0][i]
            Jx += self.contact_pairs[i_p].barycentric0[i] * x[i_v, i_b]
        for i in qd.static(range(4)):
            i_v = fem_info.elements.el2v[i_g1][i]
            Jx -= self.contact_pairs[i_p].barycentric1[i] * x[i_v, i_b]
        return qd.Vector(
            [
                Jx.dot(self.contact_pairs[i_p].tangent0),
                Jx.dot(self.contact_pairs[i_p].tangent1),
                Jx.dot(self.contact_pairs[i_p].normal),
            ]
        )

    @qd.func
    def add_Jt_x(
        self,
        i_p: int,
        x: qd.types.vector(3),
        y: qd.Tensor,
        fem_info: FEMInfo,
        vertex_constraints: FEMVertexConstraints,
        fem_config: qd.template(),
    ):
        i_b = self.contact_pairs[i_p].batch_idx
        i_g0 = self.contact_pairs[i_p].geom_idx0
        i_g1 = self.contact_pairs[i_p].geom_idx1
        world = qd.Matrix.cols(
            [self.contact_pairs[i_p].tangent0, self.contact_pairs[i_p].tangent1, self.contact_pairs[i_p].normal]
        )
        x_ = world @ x
        for i in qd.static(range(4)):
            i_v = fem_info.elements.el2v[i_g0][i]
            if qd.static(fem_config.enable_vertex_constraints):
                if not vertex_constraints.is_constrained[i_v, i_b]:
                    qd.atomic_add(y[i_v, i_b], self.contact_pairs[i_p].barycentric0[i] * x_)
            else:
                qd.atomic_add(y[i_v, i_b], self.contact_pairs[i_p].barycentric0[i] * x_)
        for i in qd.static(range(4)):
            i_v = fem_info.elements.el2v[i_g1][i]
            if qd.static(fem_config.enable_vertex_constraints):
                if not vertex_constraints.is_constrained[i_v, i_b]:
                    qd.atomic_sub(y[i_v, i_b], self.contact_pairs[i_p].barycentric1[i] * x_)
            else:
                qd.atomic_sub(y[i_v, i_b], self.contact_pairs[i_p].barycentric1[i] * x_)

    @qd.func
    def add_Jt_A_J_diag3x3(
        self,
        i_p: int,
        A: qd.types.matrix(3, 3),
        y: qd.Tensor,
        fem_info: FEMInfo,
        vertex_constraints: FEMVertexConstraints,
        fem_config: qd.template(),
    ):
        i_b = self.contact_pairs[i_p].batch_idx
        i_g0 = self.contact_pairs[i_p].geom_idx0
        i_g1 = self.contact_pairs[i_p].geom_idx1
        world = qd.Matrix.cols(
            [self.contact_pairs[i_p].tangent0, self.contact_pairs[i_p].tangent1, self.contact_pairs[i_p].normal]
        )
        B_ = world @ A @ world.transpose()
        for i in qd.static(range(4)):
            i_v = fem_info.elements.el2v[i_g0][i]
            if qd.static(fem_config.enable_vertex_constraints):
                if not vertex_constraints.is_constrained[i_v, i_b]:
                    qd.atomic_add(y[i_v, i_b], self.contact_pairs[i_p].barycentric0[i] ** 2 * B_)
            else:
                qd.atomic_add(y[i_v, i_b], self.contact_pairs[i_p].barycentric0[i] ** 2 * B_)
        for i in qd.static(range(4)):
            i_v = fem_info.elements.el2v[i_g1][i]
            if qd.static(fem_config.enable_vertex_constraints):
                if not vertex_constraints.is_constrained[i_v, i_b]:
                    qd.atomic_add(y[i_v, i_b], self.contact_pairs[i_p].barycentric1[i] ** 2 * B_)
            else:
                qd.atomic_add(y[i_v, i_b], self.contact_pairs[i_p].barycentric1[i] ** 2 * B_)

    @qd.func
    def compute_delassus(self, i_p: int, fem_state: FEMState, fem_info: FEMInfo):
        dt2_inv = fem_info.dt2_inv[None]
        i_b = self.contact_pairs[i_p].batch_idx
        i_g0 = self.contact_pairs[i_p].geom_idx0
        i_g1 = self.contact_pairs[i_p].geom_idx1
        world = qd.Matrix.cols(
            [self.contact_pairs[i_p].tangent0, self.contact_pairs[i_p].tangent1, self.contact_pairs[i_p].normal]
        )
        W = qd.Matrix.zero(gs.qd_float, 3, 3)
        # W = sum (JA^-1J^T)
        # With floor, J is Identity times the barycentric coordinates
        for i in qd.static(range(4)):
            i_v = fem_info.elements.el2v[i_g0][i]
            W += self.contact_pairs[i_p].barycentric0[i] ** 2 * dt2_inv * fem_state.vertices_solve.prec[i_v, i_b]
        for i in qd.static(range(4)):
            i_v = fem_info.elements.el2v[i_g1][i]
            W += self.contact_pairs[i_p].barycentric1[i] ** 2 * dt2_inv * fem_state.vertices_solve.prec[i_v, i_b]
        W = world.transpose() @ W @ world
        return W


@qd.data_oriented
class FEMFloorVertContactHandler(FEMContactHandler):
    """
    Class for handling contact between tetrahedral elements and a floor in a simulation using point contact model.

    This class extends the FEMContact class and provides methods for detecting contact
    between the tetrahedral elements and the floor, computing contact pairs, and managing
    contact-related computations.
    """

    def __init__(self, simulator: "Simulator") -> None:
        super().__init__()
        fem_solver = simulator.fem_solver
        self.name = "FEMFloorVertContactHandler"

        self.contact_pair_type = qd.types.struct(
            batch_idx=gs.qd_int,  # batch index
            geom_idx=gs.qd_int,  # index of the vertex
            contact_pos=gs.qd_vec3,  # contact position
            contact_info=self.contact_info_type,
        )
        self.max_contact_pairs = fem_solver.n_surface_elements * fem_solver._B
        self.contact_pairs = self.contact_pair_type.field(shape=(self.max_contact_pairs,))

    @qd.func
    def detection(
        self,
        f: int,
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
    ):
        has_overflow = False
        contact_info = qd.static(self.contact_pairs.contact_info)
        # Compute contact pairs
        self.n_contact_pairs[None] = 0
        for i_sv, i_b in qd.ndrange(
            fem_info.surface_vertices.shape[0],
            sap_state.envs.is_active.shape[0],
            axes=qd.static(eoa(fem_state.vertices.pos)),
        ):
            i_v = fem_info.surface_vertices[i_sv]
            i_g = sap_info.ground_geom_idx[None]
            pos_v = fem_state.vertices.pos[f, i_v, i_b]
            distance = pos_v.z - geoms_state.pos[i_g, i_b].z
            if distance > 0.0:
                continue
            i_p = qd.atomic_add(self.n_contact_pairs[None], 1)
            if i_p < self.max_contact_pairs:
                self.contact_pairs[i_p].batch_idx = i_b
                self.contact_pairs[i_p].geom_idx = i_v
                contact_info[i_p].k = sap_info.point_contact_stiffness[None] * fem_info.surface_vert_mass[i_sv]
                contact_info[i_p].phi0 = distance
                contact_info[i_p].mu = fem_info.vertices.friction_mu[i_v]
            else:
                has_overflow = True
        return has_overflow

    @qd.func
    def compute_Jx(self, i_p: int, x: qd.Tensor, fem_info: FEMInfo):
        """
        Compute the contact Jacobian J times a vector x.
        """
        i_b = self.contact_pairs[i_p].batch_idx
        i_g = self.contact_pairs[i_p].geom_idx
        Jx = x[i_g, i_b]
        return Jx

    @qd.func
    def add_Jt_x(
        self,
        i_p: int,
        x: qd.types.vector(3),
        y: qd.Tensor,
        fem_info: FEMInfo,
        vertex_constraints: FEMVertexConstraints,
        fem_config: qd.template(),
    ):
        i_b = self.contact_pairs[i_p].batch_idx
        i_g = self.contact_pairs[i_p].geom_idx
        if qd.static(fem_config.enable_vertex_constraints):
            if not vertex_constraints.is_constrained[i_g, i_b]:
                qd.atomic_add(y[i_g, i_b], x)
        else:
            qd.atomic_add(y[i_g, i_b], x)

    @qd.func
    def add_Jt_A_J_diag3x3(
        self,
        i_p: int,
        A: qd.types.matrix(3, 3),
        y: qd.Tensor,
        fem_info: FEMInfo,
        vertex_constraints: FEMVertexConstraints,
        fem_config: qd.template(),
    ):
        i_b = self.contact_pairs[i_p].batch_idx
        i_g = self.contact_pairs[i_p].geom_idx
        if qd.static(fem_config.enable_vertex_constraints):
            if not vertex_constraints.is_constrained[i_g, i_b]:
                qd.atomic_add(y[i_g, i_b], A)
        else:
            qd.atomic_add(y[i_g, i_b], A)

    @qd.func
    def compute_delassus(self, i_p: int, fem_state: FEMState, fem_info: FEMInfo):
        dt2_inv = fem_info.dt2_inv[None]
        i_b = self.contact_pairs[i_p].batch_idx
        i_g = self.contact_pairs[i_p].geom_idx
        # W = sum (JA^-1J^T)
        # With floor, J is Identity
        W = fem_state.vertices_solve.prec[i_g, i_b] * dt2_inv
        return W


@qd.data_oriented
class RigidFloorVertContactHandler(RigidContactHandler):
    def __init__(self, simulator: "Simulator") -> None:
        super().__init__()
        rigid_solver = simulator.rigid_solver
        self.name = "RigidFloorVertContactHandler"
        self.contact_pair_type = qd.types.struct(
            batch_idx=gs.qd_int,  # batch index
            link_idx=gs.qd_int,  # index of the link
            contact_pos=gs.qd_vec3,  # contact position
            contact_info=self.contact_info_type,
        )
        self.max_contact_pairs = rigid_solver.n_free_verts * simulator._B
        self.contact_pairs = self.contact_pair_type.field(shape=(self.max_contact_pairs,))
        self.Jt = qd.field(gs.qd_vec3, shape=(self.max_contact_pairs, rigid_solver.n_dofs_))
        self.M_inv_Jt = qd.field(gs.qd_vec3, shape=(self.max_contact_pairs, rigid_solver.n_dofs_))
        self.W = qd.field(gs.qd_mat3, shape=(self.max_contact_pairs,))

    @qd.func
    def detection(
        self,
        f: int,
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
    ):
        has_overflow = False
        contact_info = qd.static(self.contact_pairs.contact_info)
        # Compute contact pairs
        self.n_contact_pairs[None] = 0
        for i_v, i_b in qd.ndrange(
            verts_info.is_fixed.shape[0], sap_state.envs.is_active.shape[0], axes=qd.static(eoa(free_verts_state.pos))
        ):
            if verts_info.is_fixed[i_v]:
                continue
            i_fv = verts_info.verts_state_idx[i_v]
            i_ground = sap_info.ground_geom_idx[None]
            pos_v = free_verts_state.pos[i_fv, i_b]
            distance = pos_v.z - geoms_state.pos[i_ground, i_b].z
            if distance > 0.0:
                continue
            i_g = verts_info.geom_idx[i_v]
            i_l = geoms_info.link_idx[i_g]
            i_p = qd.atomic_add(self.n_contact_pairs[None], 1)
            if i_p < self.max_contact_pairs:
                self.contact_pairs[i_p].batch_idx = i_b
                self.contact_pairs[i_p].link_idx = i_l
                self.contact_pairs[i_p].contact_pos = pos_v
                contact_info[i_p].k = sap_info.rigid_floor_vert_stiffness[None]
                contact_info[i_p].phi0 = distance
                contact_info[i_p].mu = geoms_info.coup_friction[i_g]
            else:
                has_overflow = True
        return has_overflow


@qd.data_oriented
class RigidFloorTetContactHandler(RigidContactHandler):
    def __init__(self, simulator: "Simulator", n_rigid_volume_elems: int) -> None:
        super().__init__()
        rigid_solver = simulator.rigid_solver
        self.name = "RigidFloorTetContactHandler"
        self.contact_candidate_type = qd.types.struct(
            batch_idx=gs.qd_int,  # batch index
            geom_idx=gs.qd_int,  # index of the element
            intersection_code=gs.qd_int,  # intersection code for the element
            distance=gs.qd_vec4,  # distance vector for the element
        )
        self.n_contact_candidates = qd.field(gs.qd_int, shape=())
        self.max_contact_candidates = n_rigid_volume_elems * simulator._B * 8
        self.contact_candidates = self.contact_candidate_type.field(shape=(self.max_contact_candidates,))

        self.contact_pair_type = qd.types.struct(
            batch_idx=gs.qd_int,  # batch index
            link_idx=gs.qd_int,  # index of the link
            contact_pos=gs.qd_vec3,  # contact position
            contact_info=self.contact_info_type,
        )
        self.max_contact_pairs = n_rigid_volume_elems * simulator._B
        self.contact_pairs = self.contact_pair_type.field(shape=(self.max_contact_pairs,))
        self.Jt = qd.field(gs.qd_vec3, shape=(self.max_contact_pairs, rigid_solver.n_dofs_))
        self.M_inv_Jt = qd.field(gs.qd_vec3, shape=(self.max_contact_pairs, rigid_solver.n_dofs_))
        self.W = qd.field(gs.qd_mat3, shape=(self.max_contact_pairs,))

    @qd.func
    def detection(
        self,
        f: int,
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
    ):
        has_overflow = False
        candidates = qd.static(self.contact_candidates)
        # Compute contact pairs
        self.n_contact_candidates[None] = 0
        # TODO Check surface element only instead of all elements
        for i_e, i_b in qd.ndrange(
            sap_info.rigid_volume_elems.shape[0],
            sap_state.envs.is_active.shape[0],
            axes=qd.static(eoa(geoms_state.pos)),
        ):
            i_g = sap_info.rigid_volume_elems_geom_idx[i_e]
            i_l = geoms_info.link_idx[i_g]
            I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l
            if links_info.is_fixed[I_l]:
                continue
            i_ground = sap_info.ground_geom_idx[None]
            ground_pos = geoms_state.pos[i_ground, i_b]
            intersection_code = qd.int32(0)
            distance = qd.Vector.zero(gs.qd_float, 4)
            for i in qd.static(range(4)):
                i_v = sap_info.rigid_volume_elems[i_e][i]
                pos_v = sap_state.rigid_volume_verts[i_v, i_b]
                distance[i] = pos_v.z - ground_pos.z
                if distance[i] > 0.0:
                    intersection_code |= 1 << i

            # check if the element intersect with the floor
            if intersection_code != 0 and intersection_code != 15:
                i_c = qd.atomic_add(self.n_contact_candidates[None], 1)
                if i_c < self.max_contact_candidates:
                    candidates[i_c].batch_idx = i_b
                    candidates[i_c].geom_idx = i_e
                    candidates[i_c].intersection_code = intersection_code
                    candidates[i_c].distance = distance
                else:
                    has_overflow = True

        pairs = qd.static(self.contact_pairs)
        contact_info = qd.static(pairs.contact_info)
        self.n_contact_pairs[None] = 0
        # Compute pair from candidates
        result_count = qd.min(self.n_contact_candidates[None], self.max_contact_candidates)
        for i_c in range(result_count):
            candidate = candidates[i_c]
            i_b = candidate.batch_idx
            i_e = candidate.geom_idx
            intersection_code = candidate.intersection_code
            distance = candidate.distance
            intersected_edges = sap_info.marching_tets_edge_table[intersection_code]
            tet_vertices = qd.Matrix.zero(gs.qd_float, 3, 4)  # 4 vertices
            tet_pressures = qd.Vector.zero(gs.qd_float, 4)  # pressures at the vertices

            for i in qd.static(range(4)):
                i_v = sap_info.rigid_volume_elems[i_e][i]
                tet_vertices[:, i] = sap_state.rigid_volume_verts[i_v, i_b]
                tet_pressures[i] = sap_info.rigid_pressure_field[i_v]

            polygon_vertices = qd.Matrix.zero(gs.qd_float, 3, 4)  # 3 or 4 vertices
            total_area = sap_info.EPS[None]  # avoid division by zero
            total_area_weighted_centroid = qd.Vector([0.0, 0.0, 0.0])
            for i in range(4):
                if intersected_edges[i] >= 0:
                    edge = sap_info.tet_edges[intersected_edges[i]]
                    pos_v0 = tet_vertices[:, edge[0]]
                    pos_v1 = tet_vertices[:, edge[1]]
                    d_v0 = distance[edge[0]]
                    d_v1 = distance[edge[1]]
                    t = d_v0 / (d_v0 - d_v1)
                    polygon_vertices[:, i] = pos_v0 + t * (pos_v1 - pos_v0)

                    # Compute tirangle area and centroid
                    if i >= 2:
                        e1 = polygon_vertices[:, i - 1] - polygon_vertices[:, 0]
                        e2 = polygon_vertices[:, i] - polygon_vertices[:, 0]
                        area = 0.5 * e1.cross(e2).norm()
                        total_area += area
                        total_area_weighted_centroid += (
                            area * (polygon_vertices[:, 0] + polygon_vertices[:, i - 1] + polygon_vertices[:, i]) / 3.0
                        )

            centroid = total_area_weighted_centroid / total_area

            # Compute barycentric coordinates
            barycentric = tet_barycentric(centroid, tet_vertices)
            pressure = (
                barycentric[0] * tet_pressures[0]
                + barycentric[1] * tet_pressures[1]
                + barycentric[2] * tet_pressures[2]
                + barycentric[3] * tet_pressures[3]
            )

            rigid_g = sap_state.rigid_pressure_gradient[i_e, i_b].z
            g = rigid_g  # harmonic average
            rigid_k = total_area * g
            rigid_phi0 = -pressure / g
            if rigid_k < sap_info.contact_eps[None] or rigid_phi0 > sap_info.contact_eps[None]:
                continue
            i_p = qd.atomic_add(self.n_contact_pairs[None], 1)
            i_g = sap_info.rigid_volume_elems_geom_idx[i_e]
            i_l = geoms_info.link_idx[i_g]
            if i_p < self.max_contact_pairs:
                pairs[i_p].batch_idx = i_b
                pairs[i_p].link_idx = i_l
                pairs[i_p].contact_pos = centroid
                contact_info[i_p].k = rigid_k
                contact_info[i_p].phi0 = rigid_phi0
                contact_info[i_p].mu = geoms_info.coup_friction[i_g]
            else:
                has_overflow = True

        return has_overflow


@qd.data_oriented
class RigidFemTriTetContactHandler(RigidFEMContactHandler):
    """
    Class for handling self-contact between tetrahedral elements in a simulation using hydroelastic model.

    This class extends the FEMContact class and provides methods for detecting self-contact
    between tetrahedral elements, computing contact pairs, and managing contact-related computations.
    """

    def __init__(self, simulator: "Simulator") -> None:
        super().__init__()
        fem_solver, rigid_solver = simulator.fem_solver, simulator.rigid_solver
        self.name = "RigidFemTriTetContactHandler"
        self.contact_candidate_type = qd.types.struct(
            batch_idx=gs.qd_int,  # batch index
            geom_idx0=gs.qd_int,  # index of the FEM element
            geom_idx1=gs.qd_int,  # index of the Rigid Triangle
            vert_idx1=gs.qd_ivec3,  # vertex indices of the rigid triangle
            normal=gs.qd_vec3,  # contact plane normal
            x=gs.qd_vec3,  # a point on the contact plane
        )
        self.n_contact_candidates = qd.field(gs.qd_int, shape=())
        self.max_contact_candidates = max(fem_solver.n_surface_elements, rigid_solver.n_faces) * fem_solver._B * 8
        self.contact_candidates = self.contact_candidate_type.field(shape=(self.max_contact_candidates,))
        self.contact_pair_type = qd.types.struct(
            batch_idx=gs.qd_int,  # batch index
            normal=gs.qd_vec3,  # contact plane normal
            tangent0=gs.qd_vec3,  # contact plane tangent0
            tangent1=gs.qd_vec3,  # contact plane tangent1
            geom_idx0=gs.qd_int,  # index of the FEM element
            barycentric0=gs.qd_vec4,  # barycentric coordinates of the contact point in tet
            link_idx=gs.qd_int,  # index of the link
            contact_pos=gs.qd_vec3,  # contact position
            contact_info=self.contact_info_type,
        )
        self.max_contact_pairs = max(fem_solver.n_surface_elements, rigid_solver.n_faces) * fem_solver._B
        self.contact_pairs = self.contact_pair_type.field(shape=(self.max_contact_pairs,))
        self.Jt = qd.field(gs.qd_vec3, shape=(self.max_contact_pairs, rigid_solver.n_dofs_))
        self.M_inv_Jt = qd.field(gs.qd_vec3, shape=(self.max_contact_pairs, rigid_solver.n_dofs_))
        self.W = qd.field(gs.qd_mat3, shape=(self.max_contact_pairs,))

    @qd.func
    def compute_candidates(
        self,
        f: int,
        fem_state: FEMState,
        free_verts_state: VertsState,
        fixed_verts_state: VertsState,
        sap_state: SAPState,
        query_results: BVHQueryResults,
        fem_info: FEMInfo,
        faces_info: FacesInfo,
        verts_info: VertsInfo,
        sap_info: SAPInfo,
    ):
        self.n_contact_candidates[None] = 0
        has_overflow = False
        result_count = qd.min(query_results.count[0], query_results.triplets.shape[0])
        for i_r in range(result_count):
            i_b, i_t, i_sq = query_results.triplets[i_r]
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

            normal = (pos_v1 - pos_v0).cross(pos_v2 - pos_v0)
            magnitude_sqr = normal.norm_sqr()
            if magnitude_sqr < sap_info.EPS[None]:
                continue
            normal *= qd.rsqrt(magnitude_sqr)
            g0 = sap_state.fem_pressure_gradient[i_q, i_b]
            if g0.dot(normal) < sap_info.EPS[None]:
                continue

            intersection_code = qd.int32(0)
            for i in qd.static(range(4)):
                i_v = fem_info.elements.el2v[i_q][i]
                pos_v = fem_state.vertices.pos[f, i_v, i_b]
                distance = (pos_v - pos_v0).dot(normal)  # signed distance
                if distance > 0.0:
                    intersection_code |= 1 << i
            if intersection_code == 0 or intersection_code == 15:
                continue

            i_c = qd.atomic_add(self.n_contact_candidates[None], 1)
            if i_c < self.max_contact_candidates:
                self.contact_candidates[i_c].batch_idx = i_b
                self.contact_candidates[i_c].normal = normal
                self.contact_candidates[i_c].x = pos_v0
                self.contact_candidates[i_c].geom_idx0 = i_q
                self.contact_candidates[i_c].geom_idx1 = i_a
                self.contact_candidates[i_c].vert_idx1 = vert_idx1
            else:
                has_overflow = True
        return has_overflow

    @qd.func
    def compute_pairs(
        self,
        f: int,
        fem_state: FEMState,
        free_verts_state: VertsState,
        fixed_verts_state: VertsState,
        sap_state: SAPState,
        fem_info: FEMInfo,
        verts_info: VertsInfo,
        geoms_info: GeomsInfo,
        sap_info: SAPInfo,
    ):
        """
        Computes the tet triangle intersection pair and their properties.

        Intersection code reference:
        https://github.com/RobotLocomotion/drake/blob/49ab120ec6f5981484918daa821fc7101e10ebc6/geometry/proximity/mesh_intersection.cc
        """
        contact_info = qd.static(self.contact_pairs.contact_info)
        has_overflow = False
        normal_signs = qd.Vector([1.0, -1.0, 1.0, -1.0])  # make normal point outward
        self.n_contact_pairs[None] = 0
        result_count = qd.min(self.n_contact_candidates[None], self.max_contact_candidates)
        for i_c in range(result_count):
            i_b = self.contact_candidates[i_c].batch_idx
            i_e = self.contact_candidates[i_c].geom_idx0

            tri_vertices = qd.Matrix.zero(gs.qd_float, 3, 3)  # 3 vertices of the triangle
            tet_vertices = qd.Matrix.zero(gs.qd_float, 3, 4)  # 4 vertices of tet 0
            tet_pressures = qd.Vector.zero(gs.qd_float, 4)  # pressures at the vertices of tet 0
            for i in qd.static(range(3)):
                i_v = self.contact_candidates[i_c].vert_idx1[i]
                i_fv = verts_info.verts_state_idx[i_v]
                if verts_info.is_fixed[i_v]:
                    tri_vertices[:, i] = fixed_verts_state.pos[i_fv]
                else:
                    tri_vertices[:, i] = free_verts_state.pos[i_fv, i_b]
            for i in qd.static(range(4)):
                i_v = fem_info.elements.el2v[i_e][i]
                tet_vertices[:, i] = fem_state.vertices.pos[f, i_v, i_b]
                tet_pressures[i] = sap_info.fem_pressure[i_v]

            polygon_vertices = qd.Matrix.zero(gs.qd_float, 3, 7)  # maximum 7 vertices
            polygon_n_vertices = 3
            for i in qd.static(range(3)):
                polygon_vertices[:, i] = tri_vertices[:, i]
            clipped_vertices = qd.Matrix.zero(gs.qd_float, 3, 7)  # maximum 7 vertices
            clipped_n_vertices = 0
            distances = qd.Vector.zero(gs.qd_float, 7)
            for face in range(4):
                clipped_n_vertices = 0
                x = tet_vertices[:, (face + 1) % 4]
                normal = (tet_vertices[:, (face + 2) % 4] - x).cross(
                    tet_vertices[:, (face + 3) % 4] - x
                ) * normal_signs[face]
                normal /= normal.norm()

                for i in range(polygon_n_vertices):
                    distances[i] = (polygon_vertices[:, i] - x).dot(normal)

                for i in range(polygon_n_vertices):
                    j = (i + 1) % polygon_n_vertices
                    if distances[i] <= 0.0:
                        clipped_vertices[:, clipped_n_vertices] = polygon_vertices[:, i]
                        clipped_n_vertices += 1
                    if distances[i] * distances[j] < 0.0:
                        wa = distances[j] / (distances[j] - distances[i])
                        wb = 1.0 - wa
                        clipped_vertices[:, clipped_n_vertices] = (
                            wa * polygon_vertices[:, i] + wb * polygon_vertices[:, j]
                        )
                        clipped_n_vertices += 1
                polygon_n_vertices = clipped_n_vertices
                polygon_vertices = clipped_vertices

                if polygon_n_vertices < 3:
                    # If the polygon has less than 3 vertices, it is not a valid contact
                    break

            if polygon_n_vertices < 3:
                continue

            total_area = 0.0
            total_area_weighted_centroid = qd.Vector.zero(gs.qd_float, 3)
            for i in range(2, polygon_n_vertices):
                e1 = polygon_vertices[:, i - 1] - polygon_vertices[:, 0]
                e2 = polygon_vertices[:, i] - polygon_vertices[:, 0]
                area = 0.5 * e1.cross(e2).norm()
                total_area += area
                total_area_weighted_centroid += (
                    area * (polygon_vertices[:, 0] + polygon_vertices[:, i - 1] + polygon_vertices[:, i]) / 3.0
                )

            centroid = total_area_weighted_centroid / total_area
            barycentric0 = tet_barycentric(centroid, tet_vertices)
            tangent0 = (polygon_vertices[:, 0] - centroid).normalized()
            tangent1 = self.contact_candidates[i_c].normal.cross(tangent0)
            deformable_g = sap_info.hydroelastic_stiffness[None]
            rigid_g = sap_state.fem_pressure_gradient[i_e, i_b].dot(self.contact_candidates[i_c].normal)
            pressure = barycentric0.dot(tet_pressures)
            if total_area < sap_info.contact_eps[None] or rigid_g < sap_info.contact_eps[None]:
                continue
            g = rigid_g * deformable_g / (deformable_g + rigid_g)  # harmonic average
            rigid_k = total_area * g
            rigid_phi0 = -pressure / g
            i_v = self.contact_candidates[i_c].vert_idx1[0]
            i_g = verts_info.geom_idx[i_v]
            i_l = geoms_info.link_idx[i_g]
            i_p = qd.atomic_add(self.n_contact_pairs[None], 1)
            if i_p < self.max_contact_pairs:
                self.contact_pairs[i_p].batch_idx = i_b
                self.contact_pairs[i_p].normal = self.contact_candidates[i_c].normal
                self.contact_pairs[i_p].tangent0 = tangent0
                self.contact_pairs[i_p].tangent1 = tangent1
                self.contact_pairs[i_p].geom_idx0 = i_e
                self.contact_pairs[i_p].barycentric0 = barycentric0
                self.contact_pairs[i_p].link_idx = i_l
                self.contact_pairs[i_p].contact_pos = centroid
                contact_info[i_p].k = rigid_k
                contact_info[i_p].phi0 = rigid_phi0
                contact_info[i_p].mu = qd.sqrt(fem_info.elements.friction_mu[i_e] * geoms_info.coup_friction[i_g])
            else:
                has_overflow = True

        return has_overflow

    @qd.func
    def detection(
        self,
        f: int,
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
    ):
        has_overflow = False
        contact_queries_state.rigid_tri.results.count[0] = 0
        func_bvh_query_leaves(contact_queries_state.rigid_tri, func_no_filter, filter_ctx=0)
        has_overflow |= (
            contact_queries_state.rigid_tri.results.count[0] > contact_queries_state.rigid_tri.results.triplets.shape[0]
        )
        has_overflow |= self.compute_candidates(
            f,
            fem_state,
            free_verts_state,
            fixed_verts_state,
            sap_state,
            contact_queries_state.rigid_tri.results,
            fem_info,
            faces_info,
            verts_info,
            sap_info,
        )
        has_overflow |= self.compute_pairs(
            f, fem_state, free_verts_state, fixed_verts_state, sap_state, fem_info, verts_info, geoms_info, sap_info
        )
        return has_overflow

    @qd.func
    def compute_delassus_world_frame(
        self,
        fem_state: FEMState,
        fem_info: FEMInfo,
        entities_info: EntitiesInfo,
        rigid_info: RigidInfo,
        sap_info: SAPInfo,
    ):
        dt2_inv = fem_info.dt2_inv[None]
        # rigid
        RigidContactHandler.compute_delassus_world_frame(self, fem_state, fem_info, entities_info, rigid_info, sap_info)
        # fem
        barycentric0 = qd.static(self.contact_pairs.barycentric0)
        for i_p in range(self.n_contact_pairs[None]):
            i_g0 = self.contact_pairs[i_p].geom_idx0
            i_b = self.contact_pairs[i_p].batch_idx
            W = self.W[i_p]
            for i in qd.static(range(4)):
                i_v = fem_info.elements.el2v[i_g0][i]
                W = W + barycentric0[i_p][i] ** 2 * dt2_inv * fem_state.vertices_solve.prec[i_v, i_b]
            self.W[i_p] = W

    @qd.func
    def compute_delassus(self, i_p: int, fem_state: FEMState, fem_info: FEMInfo):
        world = qd.Matrix.cols(
            [self.contact_pairs[i_p].tangent0, self.contact_pairs[i_p].tangent1, self.contact_pairs[i_p].normal]
        )
        return world.transpose() @ self.W[i_p] @ world

    @qd.func
    def compute_Jx(self, i_p: int, x0: qd.Tensor, x1: qd.Tensor, fem_info: FEMInfo, sap_info: SAPInfo):
        """
        Compute the contact Jacobian J times a vector x.
        """
        i_b = self.contact_pairs[i_p].batch_idx
        i_g0 = self.contact_pairs[i_p].geom_idx0
        Jx = qd.Vector.zero(gs.qd_float, 3)

        # fem
        for i in qd.static(range(4)):
            i_v = fem_info.elements.el2v[i_g0][i]
            Jx = Jx + self.contact_pairs[i_p].barycentric0[i] * x0[i_v, i_b]

        # rigid
        for i in range(sap_info.n_dofs[None]):
            Jx = Jx - self.Jt[i_p, i] * x1[i, i_b]
        return qd.Vector(
            [
                Jx.dot(self.contact_pairs[i_p].tangent0),
                Jx.dot(self.contact_pairs[i_p].tangent1),
                Jx.dot(self.contact_pairs[i_p].normal),
            ]
        )

    @qd.func
    def add_Jt_x(
        self,
        i_p: int,
        x: qd.types.vector(3),
        y0: qd.Tensor,
        y1: qd.Tensor,
        fem_info: FEMInfo,
        sap_info: SAPInfo,
    ):
        i_b = self.contact_pairs[i_p].batch_idx
        i_g0 = self.contact_pairs[i_p].geom_idx0
        world = qd.Matrix.cols(
            [self.contact_pairs[i_p].tangent0, self.contact_pairs[i_p].tangent1, self.contact_pairs[i_p].normal]
        )
        x_ = world @ x

        # fem
        for i in qd.static(range(4)):
            i_v = fem_info.elements.el2v[i_g0][i]
            qd.atomic_add(y0[i_v, i_b], self.contact_pairs[i_p].barycentric0[i] * x_)

        # rigid
        for i in range(sap_info.n_dofs[None]):
            qd.atomic_sub(y1[i, i_b], self.Jt[i_p, i].dot(x_))

    @qd.func
    def add_Jt_A_J_diag3x3(self, i_p: int, A: qd.types.matrix(3, 3), y: qd.Tensor, fem_info: FEMInfo):
        i_b = self.contact_pairs[i_p].batch_idx
        i_g0 = self.contact_pairs[i_p].geom_idx0
        world = qd.Matrix.cols(
            [self.contact_pairs[i_p].tangent0, self.contact_pairs[i_p].tangent1, self.contact_pairs[i_p].normal]
        )
        B_ = world @ A @ world.transpose()
        for i in qd.static(range(4)):
            i_v = fem_info.elements.el2v[i_g0][i]
            if i_v < fem_info.vertices.mass.shape[0]:
                qd.atomic_add(y[i_v, i_b], self.contact_pairs[i_p].barycentric0[i] ** 2 * B_)


@qd.data_oriented
class RigidRigidTetContactHandler(RigidRigidContactHandler):
    """
    Class for handling contact between Rigid bodies using hydroelastic model.

    This class extends the RigidContact class and provides methods for detecting contact
    between tetrahedral elements, computing contact pairs, and managing contact-related computations.
    """

    def __init__(self, simulator: "Simulator", n_rigid_volume_elems: int) -> None:
        super().__init__()
        rigid_solver = simulator.rigid_solver
        self.name = "RigidRigidTetContactHandler"
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
        self.max_contact_candidates = n_rigid_volume_elems * simulator._B * 8
        self.contact_candidates = self.contact_candidate_type.field(shape=(self.max_contact_candidates,))

        self.contact_pair_type = qd.types.struct(
            batch_idx=gs.qd_int,  # batch index
            normal=gs.qd_vec3,  # contact plane normal
            tangent0=gs.qd_vec3,  # contact plane tangent0
            tangent1=gs.qd_vec3,  # contact plane tangent1
            link_idx0=gs.qd_int,  # index of the link
            link_idx1=gs.qd_int,  # index of the other link
            contact_pos=gs.qd_vec3,  # contact position
            contact_info=self.contact_info_type,
        )
        self.max_contact_pairs = n_rigid_volume_elems * simulator._B
        self.contact_pairs = self.contact_pair_type.field(shape=(self.max_contact_pairs,))
        self.Jt = qd.field(gs.qd_vec3, shape=(self.max_contact_pairs, rigid_solver.n_dofs_))
        self.M_inv_Jt = qd.field(gs.qd_vec3, shape=(self.max_contact_pairs, rigid_solver.n_dofs_))
        self.W = qd.field(gs.qd_mat3, shape=(self.max_contact_pairs,))

    @qd.func
    def compute_candidates(self, sap_state: SAPState, query_results: BVHQueryResults, sap_info: SAPInfo):
        has_overflow = False
        candidates = qd.static(self.contact_candidates)
        self.n_contact_candidates[None] = 0
        result_count = qd.min(query_results.count[0], query_results.triplets.shape[0])
        for i_r in range(result_count):
            i_b, i_a, i_q = query_results.triplets[i_r]
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
            if g0_norm < sap_info.EPS[None] or g1_norm < sap_info.EPS[None]:
                continue
            # Calculate the isosurface, i.e. equal pressure plane defined by x and normal
            # Solve for p0 + g0.dot(x - x0) = p1 + g1.dot(x - x1)
            normal = g0 - g1
            magnitude = normal.norm()
            if magnitude < sap_info.EPS[None]:
                continue
            normal /= magnitude
            b = p1 - p0 - g1.dot(x1) + g0.dot(x0)
            x = b / magnitude * normal
            # Check that the normal is pointing along g0 and against g1, some allowance as used in Drake
            if normal.dot(g0) < sap_info.contact_eps[None] or normal.dot(g1) > -sap_info.contact_eps[None]:
                continue

            intersection_code0 = qd.int32(0)
            distance0 = qd.Vector([0.0, 0.0, 0.0, 0.0])
            intersection_code1 = qd.int32(0)
            distance1 = qd.Vector([0.0, 0.0, 0.0, 0.0])
            for i in qd.static(range(4)):
                i_v = sap_info.rigid_volume_elems[i_a][i]
                pos_v = sap_state.rigid_volume_verts[i_v, i_b]
                distance0[i] = (pos_v - x).dot(normal)  # signed distance
                if distance0[i] > 0:
                    intersection_code0 |= 1 << i
            for i in qd.static(range(4)):
                i_v = sap_info.rigid_volume_elems[i_q][i]
                pos_v = sap_state.rigid_volume_verts[i_v, i_b]
                distance1[i] = (pos_v - x).dot(normal)
                if distance1[i] > 0:
                    intersection_code1 |= 1 << i
            # Fast check for whether both tets intersect with the plane
            if (
                intersection_code0 == 0
                or intersection_code1 == 0
                or intersection_code0 == 15
                or intersection_code1 == 15
            ):
                continue
            i_c = qd.atomic_add(self.n_contact_candidates[None], 1)
            if i_c < self.max_contact_candidates:
                candidates[i_c].batch_idx = i_b
                candidates[i_c].normal = normal
                candidates[i_c].x = x
                candidates[i_c].geom_idx0 = i_a
                candidates[i_c].intersection_code0 = intersection_code0
                candidates[i_c].distance0 = distance0
                candidates[i_c].geom_idx1 = i_q
            else:
                has_overflow = True
        return has_overflow

    @qd.func
    def compute_pairs(self, sap_state: SAPState, geoms_info: GeomsInfo, sap_info: SAPInfo):
        has_overflow = False
        candidates = qd.static(self.contact_candidates)
        pairs = qd.static(self.contact_pairs)
        contact_info = qd.static(pairs.contact_info)
        normal_signs = qd.Vector([1.0, -1.0, 1.0, -1.0])  # make normal point outward
        self.n_contact_pairs[None] = 0
        result_count = qd.min(self.n_contact_candidates[None], self.max_contact_candidates)
        for i_c in range(result_count):
            i_b = candidates[i_c].batch_idx
            i_e0 = candidates[i_c].geom_idx0
            i_e1 = candidates[i_c].geom_idx1
            intersection_code0 = candidates[i_c].intersection_code0
            distance0 = candidates[i_c].distance0
            intersected_edges0 = sap_info.marching_tets_edge_table[intersection_code0]
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
            clipped_vertices = qd.Matrix.zero(gs.qd_float, 3, 8)  # maximum 8 vertices
            clipped_n_vertices = gs.qd_int(0)
            for i in range(4):
                if intersected_edges0[i] >= 0:
                    edge = sap_info.tet_edges[intersected_edges0[i]]
                    pos_v0 = tet_vertices0[:, edge[0]]
                    pos_v1 = tet_vertices0[:, edge[1]]
                    d_v0 = distance0[edge[0]]
                    d_v1 = distance0[edge[1]]
                    t = d_v0 / (d_v0 - d_v1)
                    polygon_vertices[:, polygon_n_vertices] = pos_v0 + t * (pos_v1 - pos_v0)
                    polygon_n_vertices += 1
            # Intersects the polygon with the four halfspaces of the four triangles
            # of the tetrahedral element1.
            for face in range(4):
                clipped_n_vertices = 0
                x = tet_vertices1[:, (face + 1) % 4]
                normal = (tet_vertices1[:, (face + 2) % 4] - x).cross(
                    tet_vertices1[:, (face + 3) % 4] - x
                ) * normal_signs[face]
                normal /= normal.norm()

                distances = qd.Vector.zero(gs.qd_float, 8)
                for i in range(polygon_n_vertices):
                    distances[i] = (polygon_vertices[:, i] - x).dot(normal)

                for i in range(polygon_n_vertices):
                    j = (i + 1) % polygon_n_vertices
                    if distances[i] <= 0.0:
                        clipped_vertices[:, clipped_n_vertices] = polygon_vertices[:, i]
                        clipped_n_vertices += 1
                        if distances[j] > 0.0:
                            wa = distances[j] / (distances[j] - distances[i])
                            wb = 1.0 - wa
                            clipped_vertices[:, clipped_n_vertices] = (
                                wa * polygon_vertices[:, i] + wb * polygon_vertices[:, j]
                            )
                            clipped_n_vertices += 1
                    elif distances[j] <= 0.0:
                        wa = distances[j] / (distances[j] - distances[i])
                        wb = 1.0 - wa
                        clipped_vertices[:, clipped_n_vertices] = (
                            wa * polygon_vertices[:, i] + wb * polygon_vertices[:, j]
                        )
                        clipped_n_vertices += 1
                polygon_n_vertices = clipped_n_vertices
                polygon_vertices = clipped_vertices

                if polygon_n_vertices < 3:
                    # If the polygon has less than 3 vertices, it is not a valid contact
                    break

            if polygon_n_vertices < 3:
                continue

            # compute centroid and area of the polygon
            total_area = 0.0  # avoid division by zero
            total_area_weighted_centroid = qd.Vector.zero(gs.qd_float, 3)
            for i in range(2, polygon_n_vertices):
                e1 = polygon_vertices[:, i - 1] - polygon_vertices[:, 0]
                e2 = polygon_vertices[:, i] - polygon_vertices[:, 0]
                area = 0.5 * e1.cross(e2).norm()
                total_area += area
                total_area_weighted_centroid += (
                    area * (polygon_vertices[:, 0] + polygon_vertices[:, i - 1] + polygon_vertices[:, i]) / 3.0
                )

            if total_area < sap_info.contact_eps[None]:
                continue
            centroid = total_area_weighted_centroid / total_area
            tangent0 = polygon_vertices[:, 0] - centroid
            tangent0 /= tangent0.norm()
            tangent1 = candidates[i_c].normal.cross(tangent0)
            g0 = sap_state.rigid_pressure_gradient[i_e0, i_b].dot(candidates[i_c].normal)
            g1 = -sap_state.rigid_pressure_gradient[i_e1, i_b].dot(candidates[i_c].normal)
            g = 1.0 / (1.0 / g0 + 1.0 / g1)  # harmonic average, can handle infinity
            rigid_k = total_area * g
            barycentric0 = tet_barycentric(centroid, tet_vertices0)
            pressure = (
                barycentric0[0] * tet_pressures0[0]
                + barycentric0[1] * tet_pressures0[1]
                + barycentric0[2] * tet_pressures0[2]
                + barycentric0[3] * tet_pressures0[3]
            )
            rigid_phi0 = -pressure / g
            if rigid_phi0 > sap_info.contact_eps[None]:
                continue
            i_p = qd.atomic_add(self.n_contact_pairs[None], 1)
            if i_p < self.max_contact_pairs:
                pairs[i_p].batch_idx = i_b
                pairs[i_p].normal = candidates[i_c].normal
                pairs[i_p].tangent0 = tangent0
                pairs[i_p].tangent1 = tangent1
                pairs[i_p].contact_pos = centroid
                i_g0 = sap_info.rigid_volume_elems_geom_idx[i_e0]
                i_g1 = sap_info.rigid_volume_elems_geom_idx[i_e1]
                i_l0 = geoms_info.link_idx[i_g0]
                i_l1 = geoms_info.link_idx[i_g1]
                pairs[i_p].link_idx0 = i_l0
                pairs[i_p].link_idx1 = i_l1
                contact_info[i_p].k = rigid_k
                contact_info[i_p].phi0 = rigid_phi0
                contact_info[i_p].mu = qd.sqrt(geoms_info.friction[i_g0] * geoms_info.friction[i_g1])
            else:
                has_overflow = True
        return has_overflow

    @qd.func
    def detection(
        self,
        f: int,
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
    ):
        has_overflow = False
        contact_queries_state.rigid_tet.results.count[0] = 0
        func_bvh_query_leaves(
            contact_queries_state.rigid_tet,
            func_filter_rigid_tets,
            filter_ctx=(sap_info.rigid_volume_elems_geom_idx, sap_info.rigid_collision_pair_idx),
        )
        has_overflow |= (
            contact_queries_state.rigid_tet.results.count[0] > contact_queries_state.rigid_tet.results.triplets.shape[0]
        )
        has_overflow |= self.compute_candidates(sap_state, contact_queries_state.rigid_tet.results, sap_info)
        has_overflow |= self.compute_pairs(sap_state, geoms_info, sap_info)
        return has_overflow
