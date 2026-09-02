from typing import TYPE_CHECKING
import math

import igl
import numpy as np
import quadrants as qd

import genesis as gs
import genesis.utils.element as eu
import genesis.utils.array_class as array_class
import genesis.utils.geom as gu
from genesis.constants import IntEnum
from genesis.engine.bvh import AABB, LBVH, FEMSurfaceTetLBVH, RigidTetLBVH
from genesis.options.solvers import SAPCouplerOptions
from genesis.repr_base import RBC

if TYPE_CHECKING:
    from genesis.engine.simulator import Simulator

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


@qd.data_oriented
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
                f"Invalid FEM floor contact type: {options.fem_floor_contact_type}. "
                "Must be one of 'tet', 'vert', or 'none'."
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
                f"Invalid rigid floor contact type: {options.rigid_floor_contact_type}. "
                "Must be one of 'vert' or 'none'."
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

    # ------------------------------------------------------------------------------------
    # --------------------------------- Initialization -----------------------------------
    # ------------------------------------------------------------------------------------

    def build(self) -> None:
        self._B = self.sim._B
        self.contact_handlers = []
        self._enable_rigid_fem_contact &= self.rigid_solver.is_active and self.fem_solver.is_active
        self._enable_fem_self_tet_contact &= self.fem_solver.is_active

        for equality in self.rigid_solver.equalities:
            if equality.type == gs.EQUALITY_TYPE.JOINT and equality.eq_obj2id < 0:
                gs.raise_exception("SAPCoupler does not support JOINT equality constraints without `joint2`.")

        init_tet_tables = False

        if self.fem_solver.is_active:
            if self.fem_solver._use_implicit_solver is False:
                gs.raise_exception(
                    "SAPCoupler requires FEM to use implicit solver. "
                    "Please set `use_implicit_solver=True` in FEM options."
                )
            if self._fem_floor_contact_type == FEMFloorContactType.TET or self._enable_fem_self_tet_contact:
                init_tet_tables = True
                self._init_hydroelastic_fem_fields_and_info()

            if self._fem_floor_contact_type == FEMFloorContactType.TET:
                self.fem_floor_tet_contact = FEMFloorTetContactHandler(self.sim)
                self.contact_handlers.append(self.fem_floor_tet_contact)

            if self._fem_floor_contact_type == FEMFloorContactType.VERT:
                self.fem_floor_vert_contact = FEMFloorVertContactHandler(self.sim)
                self.contact_handlers.append(self.fem_floor_vert_contact)

            if self._enable_fem_self_tet_contact:
                self.fem_self_tet_contact = FEMSelfTetContactHandler(self.sim)
                self.contact_handlers.append(self.fem_self_tet_contact)

            self._init_fem_fields()

        if self.rigid_solver.is_active:
            if (
                self._rigid_floor_contact_type == RigidFloorContactType.TET
                or self._rigid_rigid_contact_type == RigidRigidContactType.TET
            ):
                init_tet_tables = True
                self._init_hydroelastic_rigid_fields_and_info()

            self._init_rigid_fields()
            if self._rigid_floor_contact_type == RigidFloorContactType.VERT:
                self.rigid_floor_vert_contact = RigidFloorVertContactHandler(self.sim)
                self.contact_handlers.append(self.rigid_floor_vert_contact)
            elif self._rigid_floor_contact_type == RigidFloorContactType.TET:
                self.rigid_floor_tet_contact = RigidFloorTetContactHandler(self.sim)
                self.contact_handlers.append(self.rigid_floor_tet_contact)

            if self._rigid_rigid_contact_type == RigidRigidContactType.TET:
                self.rigid_rigid_tet_contact = RigidRigidTetContactHandler(self.sim)
                self.contact_handlers.append(self.rigid_rigid_tet_contact)

            # TODO: Dynamically added constraints are not supported for now
            if self.rigid_solver.n_equalities > 0:
                self._init_equality_constraint()

        if self._enable_rigid_fem_contact:
            self.rigid_fem_contact = RigidFemTriTetContactHandler(self.sim)
            self.contact_handlers.append(self.rigid_fem_contact)

        self._init_bvh()
        if init_tet_tables:
            self._init_tet_tables()
        self._init_sap_fields()
        self._init_pcg_fields()
        self._init_linesearch_fields()

    def reset(self, envs_idx=None):
        pass

    def _init_tet_tables(self):
        # Lookup table for marching tetrahedra edges
        self.MarchingTetsEdgeTable = qd.field(gs.qd_ivec4, shape=len(MARCHING_TETS_EDGE_TABLE))
        self.MarchingTetsEdgeTable.from_numpy(np.array(MARCHING_TETS_EDGE_TABLE, dtype=gs.np_int))

        self.TetEdges = qd.field(gs.qd_ivec2, shape=(len(TET_EDGES),))
        self.TetEdges.from_numpy(np.array(TET_EDGES, dtype=gs.np_int))

    def _init_hydroelastic_fem_fields_and_info(self):
        self.fem_pressure = qd.field(gs.qd_float, shape=(self.fem_solver.n_vertices))
        fem_pressure_np = np.concatenate([fem_entity.pressure_field_np for fem_entity in self.fem_solver.entities])
        self.fem_pressure.from_numpy(fem_pressure_np)
        self.fem_pressure_gradient = qd.field(gs.qd_vec3, shape=(self.fem_solver._B, self.fem_solver.n_elements))

    def _init_hydroelastic_rigid_fields_and_info(self):
        rigid_volume_verts = []
        rigid_volume_elems = []
        rigid_volume_verts_geom_idx = []
        rigid_volume_elems_geom_idx = []
        rigid_pressure_field = []
        offset = 0
        for geom in self.rigid_solver.geoms:
            if geom.contype or geom.conaffinity:
                if geom.type == gs.GEOM_TYPE.PLANE:
                    gs.raise_exception("Primitive plane not supported as user-specified collision geometries.")
                volume = geom.get_trimesh().volume
                tet_cfg = {"nobisect": False, "maxvolume": volume / 100}
                mesh_verts, mesh_elems = eu.mesh_to_elements(geom.get_trimesh(), tet_cfg=tet_cfg)
                verts, elems = eu.split_all_surface_tets(mesh_verts, mesh_elems)
                rigid_volume_verts.append(verts)
                rigid_volume_elems.append(elems + offset)
                rigid_volume_verts_geom_idx.append(np.full(len(verts), geom.idx, dtype=gs.np_int))
                rigid_volume_elems_geom_idx.append(np.full(len(elems), geom.idx, dtype=gs.np_int))
                signed_distance, *_ = igl.signed_distance(verts, geom.init_verts, geom.init_faces)
                signed_distance = signed_distance.astype(gs.np_float, copy=False)

                distance_unsigned = np.abs(signed_distance)
                distance_max = np.max(distance_unsigned)
                if distance_max < gs.EPS:
                    gs.raise_exception(
                        f"Pressure field max distance is too small: {distance_max}. "
                        "This might be due to a mesh having no internal vertices."
                    )
                pressure_field_np = distance_unsigned / distance_max * self._hydroelastic_stiffness
                rigid_pressure_field.append(pressure_field_np)
                offset += len(verts)
        if not rigid_volume_verts:
            gs.raise_exception("No rigid collision geometries found.")
        rigid_volume_verts_np = np.concatenate(rigid_volume_verts, axis=0, dtype=gs.np_float)
        rigid_volume_elems_np = np.concatenate(rigid_volume_elems, axis=0, dtype=gs.np_int)
        rigid_volume_verts_geom_idx_np = np.concatenate(rigid_volume_verts_geom_idx, axis=0, dtype=gs.np_int)
        rigid_volume_elems_geom_idx_np = np.concatenate(rigid_volume_elems_geom_idx, axis=0, dtype=gs.np_int)
        rigid_pressure_field_np = np.concatenate(rigid_pressure_field, axis=0, dtype=gs.np_float)

        self.n_rigid_volume_verts = len(rigid_volume_verts_np)
        self.n_rigid_volume_elems = len(rigid_volume_elems_np)
        self.rigid_volume_verts_rest = qd.field(gs.qd_vec3, shape=(self.n_rigid_volume_verts,))
        self.rigid_volume_verts_rest.from_numpy(rigid_volume_verts_np)
        self.rigid_volume_verts = qd.field(gs.qd_vec3, shape=(self._B, self.n_rigid_volume_verts))
        self.rigid_volume_elems = qd.field(gs.qd_ivec4, shape=(self.n_rigid_volume_elems,))
        self.rigid_volume_elems.from_numpy(rigid_volume_elems_np)
        self.rigid_volume_verts_geom_idx = qd.field(gs.qd_int, shape=(self.n_rigid_volume_verts,))
        self.rigid_volume_verts_geom_idx.from_numpy(rigid_volume_verts_geom_idx_np)
        self.rigid_volume_elems_geom_idx = qd.field(gs.qd_int, shape=(self.n_rigid_volume_elems,))
        self.rigid_volume_elems_geom_idx.from_numpy(rigid_volume_elems_geom_idx_np)
        # FIXME: Convert collision_pair_idx to field here because SAPCoupler cannot support ndarray/field switch yet
        np_collision_pair_idx = self.rigid_solver.collider._collider_info.collision_pair_idx.to_numpy()
        self.rigid_collision_pair_idx = qd.field(gs.qd_int, shape=np_collision_pair_idx.shape)
        self.rigid_collision_pair_idx.from_numpy(np_collision_pair_idx)
        self.rigid_pressure_field = qd.field(gs.qd_float, shape=(self.n_rigid_volume_verts,))
        self.rigid_pressure_field.from_numpy(rigid_pressure_field_np)
        self.rigid_pressure_gradient_rest = qd.field(gs.qd_vec3, shape=(self.n_rigid_volume_elems,))
        self.rigid_pressure_gradient = qd.field(gs.qd_vec3, shape=(self._B, self.n_rigid_volume_elems))
        self.rigid_compute_pressure_gradient_rest()
        self._rigid_compliant = True

    @qd.kernel
    def rigid_update_volume_verts_pressure_gradient(self, geoms_state: array_class.GeomsState):
        for i_b, i_v in qd.ndrange(self._B, self.n_rigid_volume_verts):
            i_g = self.rigid_volume_verts_geom_idx[i_v]
            pos = geoms_state.pos[i_g, i_b]
            quat = geoms_state.quat[i_g, i_b]
            R = gu.qd_quat_to_R(quat, gs.EPS)
            self.rigid_volume_verts[i_b, i_v] = R @ self.rigid_volume_verts_rest[i_v] + pos

        for i_b, i_e in qd.ndrange(self._B, self.n_rigid_volume_elems):
            i_g = self.rigid_volume_elems_geom_idx[i_e]
            pos = geoms_state.pos[i_g, i_b]
            quat = geoms_state.quat[i_g, i_b]
            R = gu.qd_quat_to_R(quat, gs.EPS)
            self.rigid_pressure_gradient[i_b, i_e] = R @ self.rigid_pressure_gradient_rest[i_e]

    @qd.kernel
    def rigid_compute_pressure_gradient_rest(self):
        grad = qd.static(self.rigid_pressure_gradient_rest)
        for i_e in range(self.n_rigid_volume_elems):
            grad[i_e].fill(0.0)
            for i in qd.static(range(4)):
                i_v0 = self.rigid_volume_elems[i_e][i]
                i_v1 = self.rigid_volume_elems[i_e][(i + 1) % 4]
                i_v2 = self.rigid_volume_elems[i_e][(i + 2) % 4]
                i_v3 = self.rigid_volume_elems[i_e][(i + 3) % 4]
                pos_v0 = self.rigid_volume_verts_rest[i_v0]
                pos_v1 = self.rigid_volume_verts_rest[i_v1]
                pos_v2 = self.rigid_volume_verts_rest[i_v2]
                pos_v3 = self.rigid_volume_verts_rest[i_v3]

                e10 = pos_v0 - pos_v1
                e12 = pos_v2 - pos_v1
                e13 = pos_v3 - pos_v1

                area_vector = e12.cross(e13)
                signed_volume = area_vector.dot(e10)
                if qd.abs(signed_volume) > gs.EPS:
                    grad_i = area_vector / signed_volume
                    grad[i_e] += grad_i * self.rigid_pressure_field[i_v0]

    def _init_bvh(self):
        if self._enable_fem_self_tet_contact:
            self.fem_surface_tet_aabb = AABB(self.fem_solver._B, self.fem_solver.n_surface_elements)
            self.fem_surface_tet_bvh = FEMSurfaceTetLBVH(
                self.fem_solver, self.fem_surface_tet_aabb, max_n_query_result_per_aabb=MAX_N_QUERY_RESULT_PER_AABB
            )

        if self._enable_rigid_fem_contact:
            self.rigid_tri_aabb = AABB(self.sim._B, self.rigid_solver.n_faces)
            max_n_query_result_per_aabb = (
                max(self.rigid_solver.n_faces, self.fem_solver.n_surface_elements)
                * MAX_N_QUERY_RESULT_PER_AABB
                // self.rigid_solver.n_faces
            )
            self.rigid_tri_bvh = LBVH(self.rigid_tri_aabb, max_n_query_result_per_aabb)

        if self.rigid_solver.is_active and self._rigid_rigid_contact_type == RigidRigidContactType.TET:
            self.rigid_tet_aabb = AABB(self.sim._B, self.n_rigid_volume_elems)
            self.rigid_tet_bvh = RigidTetLBVH(
                self, self.rigid_tet_aabb, max_n_query_result_per_aabb=MAX_N_QUERY_RESULT_PER_AABB
            )

    def _init_equality_constraint(self):
        # TODO: Handling dynamically registered weld constraints would requiere passing 'constraint_state' as input.
        # This is not a big deal for now since only joint equality constraints are support by this coupler.
        self.equality_constraint_handler = RigidConstraintHandler(self.sim)
        self.equality_constraint_handler.build_constraints(
            self.rigid_solver.dyn_info.equalities, self.rigid_solver.dyn_info.joints, self.rigid_solver.rigid_config
        )

    def _init_sap_fields(self):
        self.batch_active = qd.field(dtype=gs.qd_bool, shape=(self.sim._B,), needs_grad=False)
        sap_state = qd.types.struct(
            gradient_norm=gs.qd_float,  # norm of the gradient
            momentum_norm=gs.qd_float,  # norm of the momentum
            impulse_norm=gs.qd_float,  # norm of the impulse
        )
        self.sap_state = sap_state.field(shape=(self.sim._B,), needs_grad=False, layout=qd.Layout.SOA)

    def _init_fem_fields(self):
        fem_state_v = qd.types.struct(
            v=gs.qd_vec3,  # vertex velocity
            v_diff=gs.qd_vec3,  # difference between current and previous velocity
            gradient=gs.qd_vec3,  # gradient vector
            impulse=gs.qd_vec3,  # impulse vector
        )

        self.fem_state_v = fem_state_v.field(
            shape=(self.sim._B, self.fem_solver.n_vertices), needs_grad=False, layout=qd.Layout.SOA
        )

        pcg_fem_state_v = qd.types.struct(
            diag3x3=gs.qd_mat3,  # diagonal 3-by-3 block of the hessian
            prec=gs.qd_mat3,  # preconditioner
            x=gs.qd_vec3,  # solution vector
            r=gs.qd_vec3,  # residual vector
            z=gs.qd_vec3,  # preconditioned residual vector
            p=gs.qd_vec3,  # search direction vector
            Ap=gs.qd_vec3,  # matrix-vector product
        )

        self.pcg_fem_state_v = pcg_fem_state_v.field(
            shape=(self.sim._B, self.fem_solver.n_vertices), needs_grad=False, layout=qd.Layout.SOA
        )

        linesearch_fem_state_v = qd.types.struct(
            x_prev=gs.qd_vec3,  # solution vector
            dp=gs.qd_vec3,  # A @ dv
        )

        self.linesearch_fem_state_v = linesearch_fem_state_v.field(
            shape=(self.sim._B, self.fem_solver.n_vertices), needs_grad=False, layout=qd.Layout.SOA
        )

    def _init_rigid_fields(self):
        rigid_state_dof = qd.types.struct(
            v=gs.qd_float,  # vertex velocity
            v_diff=gs.qd_float,  # difference between current and previous velocity
            mass_v_diff=gs.qd_float,  # mass weighted difference between current and previous velocity
            gradient=gs.qd_float,  # gradient vector
            impulse=gs.qd_float,  # impulse vector
        )

        self.rigid_state_dof = rigid_state_dof.field(
            shape=(self.sim._B, self.rigid_solver.n_dofs), needs_grad=False, layout=qd.Layout.SOA
        )

        pcg_rigid_state_dof = qd.types.struct(
            x=gs.qd_float,  # solution vector
            r=gs.qd_float,  # residual vector
            z=gs.qd_float,  # preconditioned residual vector
            p=gs.qd_float,  # search direction vector
            Ap=gs.qd_float,  # matrix-vector product
        )

        self.pcg_rigid_state_dof = pcg_rigid_state_dof.field(
            shape=(self.sim._B, self.rigid_solver.n_dofs), needs_grad=False, layout=qd.Layout.SOA
        )

        linesearch_rigid_state_dof = qd.types.struct(
            x_prev=gs.qd_float,  # solution vector
            dp=gs.qd_float,  # A @ dv
        )
        self.linesearch_rigid_state_dof = linesearch_rigid_state_dof.field(
            shape=(self.sim._B, self.rigid_solver.n_dofs), needs_grad=False, layout=qd.Layout.SOA
        )

    def _init_pcg_fields(self):
        self.batch_pcg_active = qd.field(dtype=gs.qd_bool, shape=(self.sim._B,), needs_grad=False)

        pcg_state = qd.types.struct(
            rTr=gs.qd_float,
            rTz=gs.qd_float,
            rTr_new=gs.qd_float,
            rTz_new=gs.qd_float,
            pTAp=gs.qd_float,
            alpha=gs.qd_float,
            beta=gs.qd_float,
        )

        self.pcg_state = pcg_state.field(shape=(self.sim._B,), needs_grad=False, layout=qd.Layout.SOA)

    def _init_linesearch_fields(self):
        self.batch_linesearch_active = qd.field(dtype=gs.qd_bool, shape=(self.sim._B,), needs_grad=False)

        linesearch_state = qd.types.struct(
            prev_energy=gs.qd_float,
            energy=gs.qd_float,
            step_size=gs.qd_float,
            m=gs.qd_float,
            dell_dalpha=gs.qd_float,  # first derivative of the total energy w.r.t. alpha
            d2ellA_dalpha2=gs.qd_float,  # second derivative of the dynamic energy w.r.t. alpha
            d2ell_dalpha2=gs.qd_float,  # second derivative of the total energy w.r.t. alpha
            dell_scale=gs.qd_float,  # scale factor for the first derivative
            alpha_min=gs.qd_float,  # minimum stepsize value
            alpha_max=gs.qd_float,  # maximum stepsize value
            alpha_tol=gs.qd_float,  # stepsize tolerance for convergence
            f_lower=gs.qd_float,  # minimum f value
            f_upper=gs.qd_float,  # maximum f value
            f=gs.qd_float,  # f value
            df=gs.qd_float,  # f gradient
            minus_dalpha=gs.qd_float,  # negative stepsize
            minus_dalpha_prev=gs.qd_float,  # previous negative stepsize
        )

        self.linesearch_state = linesearch_state.field(shape=(self.sim._B,), needs_grad=False, layout=qd.Layout.SOA)

    # ------------------------------------------------------------------------------------
    # -------------------------------------- Main ----------------------------------------
    # ------------------------------------------------------------------------------------

    def preprocess(self, i_step):
        self.precompute(i_step)
        self.update_bvh(i_step)
        self.has_contact, overflow = self.update_contact(
            i_step,
            links_info=self.rigid_solver.dyn_info.links,
            faces_info=self.rigid_solver.dyn_info.faces,
            verts_info=self.rigid_solver.dyn_info.verts,
            free_verts_state=self.rigid_solver.dyn_state.free_verts,
            fixed_verts_state=self.rigid_solver.dyn_state.fixed_verts,
            geoms_info=self.rigid_solver.dyn_info.geoms,
            dofs_state=self.rigid_solver.dyn_state.dofs,
            links_state=self.rigid_solver.dyn_state.links,
        )
        if overflow:
            message = "Overflowed In Contact Query: \n"
            for contact in self.contact_handlers:
                if contact.n_contact_pairs[None] > contact.max_contact_pairs:
                    message += (
                        f"{contact.name} max contact pairs: {contact.max_contact_pairs}"
                        f", using {contact.n_contact_pairs[None]}\n"
                    )
            gs.raise_exception(message)
        self.compute_regularization(
            dofs_state=self.rigid_solver.dyn_state.dofs,
            entities_info=self.rigid_solver.dyn_info.entities,
            rigid_info=self.rigid_solver.rigid_info,
        )

    def precompute(self, i_step):
        from genesis.engine.solvers.rigid.rigid_solver import kernel_update_all_verts

        if self.fem_solver.is_active:
            if qd.static(self._fem_floor_contact_type == FEMFloorContactType.TET or self._enable_fem_self_tet_contact):
                self.fem_compute_pressure_gradient(i_step)

        if self.rigid_solver.is_active:
            kernel_update_all_verts(
                self.rigid_solver.dyn_state, self.rigid_solver.dyn_info, self.rigid_solver.rigid_config
            )

        if self._rigid_compliant:
            self.rigid_update_volume_verts_pressure_gradient(self.rigid_solver.dyn_state.geoms)

    @qd.kernel
    def update_contact(
        self,
        i_step: qd.i32,
        links_info: array_class.LinksInfo,
        faces_info: array_class.FacesInfo,
        verts_info: array_class.VertsInfo,
        free_verts_state: array_class.VertsState,
        fixed_verts_state: array_class.VertsState,
        geoms_info: array_class.GeomsInfo,
        dofs_state: array_class.DofsState,
        links_state: array_class.LinksState,
    ) -> tuple[bool, bool]:
        has_contact = False
        overflow = False
        for contact in qd.static(self.contact_handlers):
            overflow |= contact.detection(
                i_step,
                links_info=links_info,
                verts_info=verts_info,
                faces_info=faces_info,
                free_verts_state=free_verts_state,
                fixed_verts_state=fixed_verts_state,
                geoms_info=geoms_info,
            )
            has_contact |= contact.n_contact_pairs[None] > 0
            contact.compute_jacobian(links_info=links_info, dofs_state=dofs_state, links_state=links_state)
        return has_contact, overflow

    def couple(self, i_step):
        if self.has_contact:
            self.sap_solve(i_step)
            self.update_vel(i_step, dofs_state=self.rigid_solver.dyn_state.dofs)

    def couple_grad(self, i_step):
        gs.raise_exception("couple_grad is not available for SAPCoupler. Please use LegacyCoupler instead.")

    @qd.kernel
    def update_vel(self, i_step: qd.i32, dofs_state: array_class.DofsState):
        if qd.static(self.fem_solver.is_active):
            self.update_fem_vel(i_step)
        if qd.static(self.rigid_solver.is_active):
            self.update_rigid_vel(dofs_state=dofs_state)

    @qd.func
    def update_fem_vel(self, i_step: qd.i32):
        for i_b, i_v in qd.ndrange(self.fem_solver._B, self.fem_solver.n_vertices):
            self.fem_solver.elements_v[i_step + 1, i_v, i_b].vel = self.fem_state_v.v[i_b, i_v]

    @qd.func
    def update_rigid_vel(self, dofs_state: array_class.DofsState):
        for i_b, i_d in qd.ndrange(self.rigid_solver._B, self.rigid_solver.n_dofs):
            dofs_state.vel[i_d, i_b] = self.rigid_state_dof.v[i_b, i_d]

    @qd.kernel
    def fem_compute_pressure_gradient(self, i_step: qd.i32):
        for i_b, i_e in qd.ndrange(self.fem_solver._B, self.fem_solver.n_elements):
            self.fem_pressure_gradient[i_b, i_e].fill(0.0)

            for i in qd.static(range(4)):
                i_v0 = self.fem_solver.elements_i[i_e].el2v[i]
                i_v1 = self.fem_solver.elements_i[i_e].el2v[(i + 1) % 4]
                i_v2 = self.fem_solver.elements_i[i_e].el2v[(i + 2) % 4]
                i_v3 = self.fem_solver.elements_i[i_e].el2v[(i + 3) % 4]
                pos_v0 = self.fem_solver.elements_v[i_step, i_v0, i_b].pos
                pos_v1 = self.fem_solver.elements_v[i_step, i_v1, i_b].pos
                pos_v2 = self.fem_solver.elements_v[i_step, i_v2, i_b].pos
                pos_v3 = self.fem_solver.elements_v[i_step, i_v3, i_b].pos

                e10 = pos_v0 - pos_v1
                e12 = pos_v2 - pos_v1
                e13 = pos_v3 - pos_v1

                area_vector = e12.cross(e13)
                signed_volume = area_vector.dot(e10)
                if qd.abs(signed_volume) > gs.EPS:
                    grad_i = area_vector / signed_volume
                    self.fem_pressure_gradient[i_b, i_e] += grad_i * self.fem_pressure[i_v0]

    # ------------------------------------------------------------------------------------
    # -------------------------------------- BVH -----------------------------------------
    # ------------------------------------------------------------------------------------

    def update_bvh(self, i_step: qd.i32):
        if self._enable_fem_self_tet_contact:
            self.update_fem_surface_tet_bvh(i_step)

        if self._enable_rigid_fem_contact:
            self.update_rigid_tri_bvh()

        if self.rigid_solver.is_active and self._rigid_rigid_contact_type == RigidRigidContactType.TET:
            self.update_rigid_tet_bvh()

    def update_fem_surface_tet_bvh(self, i_step: qd.i32):
        self.compute_fem_surface_tet_aabb(i_step)
        self.fem_surface_tet_bvh.build()

    def update_rigid_tri_bvh(self):
        self.compute_rigid_tri_aabb(
            faces_info=self.rigid_solver.dyn_info.faces,
            free_verts_state=self.rigid_solver.dyn_state.free_verts,
            fixed_verts_state=self.rigid_solver.dyn_state.fixed_verts,
            verts_info=self.rigid_solver.dyn_info.verts,
        )
        self.rigid_tri_bvh.build()

    def update_rigid_tet_bvh(self):
        self.compute_rigid_tet_aabb()
        self.rigid_tet_bvh.build()

    @qd.kernel
    def compute_fem_surface_tet_aabb(self, i_step: qd.i32):
        aabbs = qd.static(self.fem_surface_tet_aabb.aabbs)
        for i_b, i_se in qd.ndrange(self.fem_solver._B, self.fem_solver.n_surface_elements):
            i_e = self.fem_solver.surface_elements[i_se]
            i_vs = self.fem_solver.elements_i[i_e].el2v

            aabbs[i_b, i_se].min.fill(np.inf)
            aabbs[i_b, i_se].max.fill(-np.inf)
            for i in qd.static(range(4)):
                pos_v = self.fem_solver.elements_v[i_step, i_vs[i], i_b].pos
                aabbs[i_b, i_se].min = qd.min(aabbs[i_b, i_se].min, pos_v)
                aabbs[i_b, i_se].max = qd.max(aabbs[i_b, i_se].max, pos_v)

    @qd.kernel
    def compute_rigid_tri_aabb(
        self,
        faces_info: array_class.FacesInfo,
        free_verts_state: array_class.VertsState,
        fixed_verts_state: array_class.VertsState,
        verts_info: array_class.VertsInfo,
    ):
        aabbs = qd.static(self.rigid_tri_aabb.aabbs)
        for i_b, i_f in qd.ndrange(self.rigid_solver._B, self.rigid_solver.n_faces):
            tri_vertices = qd.Matrix.zero(gs.qd_float, 3, 3)
            for i in qd.static(range(3)):
                i_v = faces_info.verts_idx[i_f][i]
                i_fv = verts_info.verts_state_idx[i_v]
                if verts_info.is_fixed[i_v]:
                    tri_vertices[:, i] = fixed_verts_state.pos[i_fv]
                else:
                    tri_vertices[:, i] = free_verts_state.pos[i_fv, i_b]
            pos_v0, pos_v1, pos_v2 = tri_vertices[:, 0], tri_vertices[:, 1], tri_vertices[:, 2]

            aabbs[i_b, i_f].min = qd.min(pos_v0, pos_v1, pos_v2)
            aabbs[i_b, i_f].max = qd.max(pos_v0, pos_v1, pos_v2)

    @qd.kernel
    def compute_rigid_tet_aabb(self):
        aabbs = qd.static(self.rigid_tet_aabb.aabbs)
        for i_b, i_e in qd.ndrange(self._B, self.n_rigid_volume_elems):
            i_v0 = self.rigid_volume_elems[i_e][0]
            i_v1 = self.rigid_volume_elems[i_e][1]
            i_v2 = self.rigid_volume_elems[i_e][2]
            i_v3 = self.rigid_volume_elems[i_e][3]
            pos_v0 = self.rigid_volume_verts[i_b, i_v0]
            pos_v1 = self.rigid_volume_verts[i_b, i_v1]
            pos_v2 = self.rigid_volume_verts[i_b, i_v2]
            pos_v3 = self.rigid_volume_verts[i_b, i_v3]
            aabbs[i_b, i_e].min = qd.min(pos_v0, pos_v1, pos_v2, pos_v3)
            aabbs[i_b, i_e].max = qd.max(pos_v0, pos_v1, pos_v2, pos_v3)

    # ------------------------------------------------------------------------------------
    # ------------------------------------- Solve ----------------------------------------
    # ------------------------------------------------------------------------------------

    def sap_solve(self, i_step):
        self._init_sap_solve(i_step, dofs_state=self.rigid_solver.dyn_state.dofs)
        for iter in range(self._n_sap_iterations):
            # init gradient and preconditioner
            self.compute_unconstrained_gradient_diag(i_step, iter)

            # compute contact hessian and gradient
            self.compute_constraint_contact_gradient_hessian_diag_prec()
            self.check_sap_convergence(rigid_info=self.rigid_solver.rigid_info)
            # solve for the vertex velocity
            self.pcg_solve()

            # line search
            self.exact_linesearch(i_step)

    @qd.kernel
    def check_sap_convergence(self, rigid_info: array_class.RigidInfo):
        self.clear_sap_norms()
        if qd.static(self.fem_solver.is_active):
            self.add_fem_norms()
        if qd.static(self.rigid_solver.is_active):
            self.add_rigid_norms(rigid_info=rigid_info)
        self.update_batch_active()

    @qd.func
    def clear_sap_norms(self):
        for i_b in range(self._B):
            if not self.batch_active[i_b]:
                continue
            self.sap_state[i_b].gradient_norm = 0.0
            self.sap_state[i_b].momentum_norm = 0.0
            self.sap_state[i_b].impulse_norm = 0.0

    @qd.func
    def add_fem_norms(self):
        for i_b, i_v in qd.ndrange(self._B, self.fem_solver.n_vertices):
            if not self.batch_active[i_b]:
                continue
            self.sap_state[i_b].gradient_norm += (
                self.fem_state_v.gradient[i_b, i_v].norm_sqr() / self.fem_solver.elements_v_info[i_v].mass
            )
            self.sap_state[i_b].momentum_norm += (
                self.fem_state_v.v[i_b, i_v].norm_sqr() * self.fem_solver.elements_v_info[i_v].mass
            )
            self.sap_state[i_b].impulse_norm += (
                self.fem_state_v.impulse[i_b, i_v].norm_sqr() / self.fem_solver.elements_v_info[i_v].mass
            )

    @qd.func
    def add_rigid_norms(self, rigid_info: array_class.RigidInfo):
        for i_b, i_d in qd.ndrange(self._B, self.rigid_solver.n_dofs):
            if not self.batch_active[i_b]:
                continue
            self.sap_state[i_b].gradient_norm += (
                self.rigid_state_dof.gradient[i_b, i_d] ** 2 / rigid_info.mass_mat[i_d, i_d, i_b]
            )
            self.sap_state[i_b].momentum_norm += (
                self.rigid_state_dof.v[i_b, i_d] ** 2 * rigid_info.mass_mat[i_d, i_d, i_b]
            )
            self.sap_state[i_b].impulse_norm += (
                self.rigid_state_dof.impulse[i_b, i_d] ** 2 / rigid_info.mass_mat[i_d, i_d, i_b]
            )

    @qd.func
    def update_batch_active(self):
        for i_b in range(self._B):
            if not self.batch_active[i_b]:
                continue
            norm_thr = self._sap_convergence_atol + self._sap_convergence_rtol * qd.max(
                self.sap_state[i_b].momentum_norm, self.sap_state[i_b].impulse_norm
            )
            self.batch_active[i_b] = self.sap_state[i_b].gradient_norm >= norm_thr

    @qd.kernel
    def compute_regularization(
        self,
        dofs_state: array_class.DofsState,
        entities_info: array_class.EntitiesInfo,
        rigid_info: array_class.RigidInfo,
    ):
        for contact in qd.static(self.contact_handlers):
            contact.compute_regularization(entities_info=entities_info, rigid_info=rigid_info)
        if qd.static(self.rigid_solver.is_active and self.rigid_solver.n_equalities > 0):
            self.equality_constraint_handler.compute_regularization(dofs_state=dofs_state)

    @qd.kernel
    def _init_sap_solve(self, i_step: qd.i32, dofs_state: array_class.DofsState):
        self._init_v(i_step, dofs_state=dofs_state)
        self.batch_active.fill(True)

    @qd.func
    def _init_v(self, i_step: qd.i32, dofs_state: array_class.DofsState):
        if qd.static(self.fem_solver.is_active):
            self._init_v_fem(i_step)
        if qd.static(self.rigid_solver.is_active):
            self._init_v_rigid(i_step, dofs_state=dofs_state)

    @qd.func
    def _init_v_fem(self, i_step: qd.i32):
        for i_b, i_v in qd.ndrange(self._B, self.fem_solver.n_vertices):
            self.fem_state_v.v[i_b, i_v] = self.fem_solver.elements_v[i_step + 1, i_v, i_b].vel

    @qd.func
    def _init_v_rigid(self, i_step: qd.i32, dofs_state: array_class.DofsState):
        for i_b, i_d in qd.ndrange(self.rigid_solver._B, self.rigid_solver.n_dofs):
            self.rigid_state_dof.v[i_b, i_d] = dofs_state.vel[i_d, i_b]

    def compute_unconstrained_gradient_diag(self, i_step: qd.i32, iter: int):
        self.init_unconstrained_gradient_diag(i_step)
        # No need to do this for iter=0 because v=v* and A(v-v*) = 0
        if iter > 0:
            self.compute_unconstrained_gradient()

    def init_unconstrained_gradient_diag(self, i_step: qd.i32):
        if self.fem_solver.is_active:
            self.init_fem_unconstrained_gradient_diag(i_step)
        if self.rigid_solver.is_active:
            self.init_rigid_unconstrained_gradient(dofs_state=self.rigid_solver.dyn_state.dofs)

    @qd.kernel
    def init_fem_unconstrained_gradient_diag(self, i_step: qd.i32):
        dt2 = self.fem_solver._substep_dt**2
        for i_b, i_v in qd.ndrange(self.fem_solver._B, self.fem_solver.n_vertices):
            self.fem_state_v.gradient[i_b, i_v].fill(0.0)
            # was using position now using velocity, need to multiply dt^2
            self.pcg_fem_state_v[i_b, i_v].diag3x3 = self.fem_solver.pcg_state_v[i_b, i_v].diag3x3 * dt2
            self.fem_state_v.v_diff[i_b, i_v] = (
                self.fem_state_v.v[i_b, i_v] - self.fem_solver.elements_v[i_step + 1, i_v, i_b].vel
            )

    @qd.kernel
    def init_rigid_unconstrained_gradient(self, dofs_state: array_class.DofsState):
        for i_b, i_d in qd.ndrange(self.rigid_solver._B, self.rigid_solver.n_dofs):
            self.rigid_state_dof.gradient[i_b, i_d] = 0.0
            self.rigid_state_dof.v_diff[i_b, i_d] = self.rigid_state_dof.v[i_b, i_d] - dofs_state.vel[i_d, i_b]

    def compute_unconstrained_gradient(self):
        if self.fem_solver.is_active:
            self.compute_fem_unconstrained_gradient()
        if self.rigid_solver.is_active:
            self.compute_rigid_unconstrained_gradient(rigid_info=self.rigid_solver.rigid_info)

    @qd.kernel
    def compute_fem_unconstrained_gradient(self):
        self.compute_fem_matrix_vector_product(self.fem_state_v.v_diff, self.fem_state_v.gradient, self.batch_active)

    @qd.kernel
    def compute_rigid_unconstrained_gradient(self, rigid_info: array_class.RigidInfo):
        self.pcg_rigid_state_dof.Ap.fill(0.0)
        for i_b, i_d0, i_d1 in qd.ndrange(self.rigid_solver._B, self.rigid_solver.n_dofs, self.rigid_solver.n_dofs):
            if not self.batch_active[i_b]:
                continue
            self.rigid_state_dof.gradient[i_b, i_d1] += (
                rigid_info.mass_mat[i_d1, i_d0, i_b] * self.rigid_state_dof.v_diff[i_b, i_d0]
            )

    @qd.kernel
    def compute_constraint_contact_gradient_hessian_diag_prec(self):
        self.clear_impulses()
        if qd.static(self.rigid_solver.is_active and self.rigid_solver.n_equalities > 0):
            self.equality_constraint_handler.compute_gradient_hessian_diag()
        for contact in qd.static(self.contact_handlers):
            contact.compute_gradient_hessian_diag()
        self.compute_preconditioner()

    @qd.func
    def clear_impulses(self):
        if qd.static(self.fem_solver.is_active):
            self.clear_fem_impulses()
        if qd.static(self.rigid_solver.is_active):
            self.clear_rigid_impulses()

    @qd.func
    def clear_fem_impulses(self):
        for i_b, i_v in qd.ndrange(self.fem_solver._B, self.fem_solver.n_vertices):
            if not self.batch_active[i_b]:
                continue
            self.fem_state_v[i_b, i_v].impulse.fill(0.0)

    @qd.func
    def clear_rigid_impulses(self):
        for i_b, i_d in qd.ndrange(self.rigid_solver._B, self.rigid_solver.n_dofs):
            if not self.batch_active[i_b]:
                continue
            self.rigid_state_dof[i_b, i_d].impulse = 0.0

    @qd.func
    def compute_preconditioner(self):
        if qd.static(self.fem_solver.is_active):
            self.compute_fem_preconditioner()

    @qd.func
    def compute_fem_preconditioner(self):
        for i_b, i_v in qd.ndrange(self.fem_solver._B, self.fem_solver.n_vertices):
            if not self.batch_active[i_b]:
                continue
            self.pcg_fem_state_v[i_b, i_v].prec = self.pcg_fem_state_v[i_b, i_v].diag3x3.inverse()

    @qd.func
    def compute_fem_pcg_matrix_vector_product(self):
        self.compute_fem_matrix_vector_product(self.pcg_fem_state_v.p, self.pcg_fem_state_v.Ap, self.batch_pcg_active)

    @qd.func
    def compute_rigid_pcg_matrix_vector_product(self, rigid_info: array_class.RigidInfo):
        self.compute_rigid_mass_mat_vec_product(
            self.pcg_rigid_state_dof.p, self.pcg_rigid_state_dof.Ap, self.batch_pcg_active, rigid_info=rigid_info
        )

    @qd.func
    def compute_elastic_products(self, i_b, i_e, S, i_vs, src):
        p9 = qd.Vector.zero(gs.qd_float, 9)
        for i, j in qd.static(qd.ndrange(3, 4)):
            p9[i * 3 : i * 3 + 3] = p9[i * 3 : i * 3 + 3] + (S[j, i] * src[i_b, i_vs[j]])

        H9_p9 = qd.Vector.zero(gs.qd_float, 9)

        for i, j in qd.static(qd.ndrange(3, 3)):
            H9_p9[i * 3 : i * 3 + 3] = H9_p9[i * 3 : i * 3 + 3] + (
                self.fem_solver.elements_el_hessian[i_b, i, j, i_e] @ p9[j * 3 : j * 3 + 3]
            )
        return p9, H9_p9

    @qd.func
    def compute_fem_matrix_vector_product(self, src, dst, active):
        """
        Compute the FEM matrix-vector product, including mass matrix and elasticity stiffness matrix.
        """
        dt2 = self.fem_solver._substep_dt**2
        damping_alpha_factor = self.fem_solver._damping_alpha * self.fem_solver._substep_dt + 1.0
        damping_beta_factor = self.fem_solver._damping_beta / self.fem_solver._substep_dt + 1.0

        # Inerita
        for i_b, i_v in qd.ndrange(self.fem_solver._B, self.fem_solver.n_vertices):
            if not active[i_b]:
                continue
            dst[i_b, i_v] = (
                self.fem_solver.elements_v_info[i_v].mass_over_dt2 * src[i_b, i_v] * dt2 * damping_alpha_factor
            )

        # Elasticity
        for i_b, i_e in qd.ndrange(self.fem_solver._B, self.fem_solver.n_elements):
            if not active[i_b]:
                continue
            V_dt2 = self.fem_solver.elements_i[i_e].V * dt2
            B = self.fem_solver.elements_i[i_e].B
            S = qd.Matrix.zero(gs.qd_float, 4, 3)
            S[:3, :] = B
            S[3, :] = -B[0, :] - B[1, :] - B[2, :]
            i_vs = self.fem_solver.elements_i[i_e].el2v

            if qd.static(self.fem_solver._enable_vertex_constraints):
                for i in qd.static(range(4)):
                    if self.fem_solver.vertex_constraints.is_constrained[i_vs[i], i_b]:
                        S[i, :] = qd.Vector.zero(gs.qd_float, 3)

            _, new_p9 = self.compute_elastic_products(i_b, i_e, S, i_vs, src)
            # atomic
            scale = V_dt2 * damping_beta_factor
            for i in qd.static(range(4)):
                dst[i_b, i_vs[i]] += (S[i, 0] * new_p9[0:3] + S[i, 1] * new_p9[3:6] + S[i, 2] * new_p9[6:9]) * scale

    @qd.kernel
    def init_pcg_solve(self, entities_info: array_class.EntitiesInfo, rigid_info: array_class.RigidInfo):
        self.init_pcg_state()
        if qd.static(self.fem_solver.is_active):
            self.init_fem_pcg_solve()
        if qd.static(self.rigid_solver.is_active):
            self.init_rigid_pcg_solve(entities_info=entities_info, rigid_info=rigid_info)
        self.init_pcg_active()

    @qd.func
    def init_pcg_state(self):
        for i_b in qd.ndrange(self._B):
            self.batch_pcg_active[i_b] = self.batch_active[i_b]
            if not self.batch_pcg_active[i_b]:
                continue
            self.pcg_state[i_b].rTr = 0.0
            self.pcg_state[i_b].rTz = 0.0

    @qd.func
    def init_fem_pcg_solve(self):
        for i_b, i_v in qd.ndrange(self._B, self.fem_solver.n_vertices):
            if not self.batch_pcg_active[i_b]:
                continue
            self.pcg_fem_state_v[i_b, i_v].x = 0.0
            self.pcg_fem_state_v[i_b, i_v].r = -self.fem_state_v.gradient[i_b, i_v]
            self.pcg_fem_state_v[i_b, i_v].z = self.pcg_fem_state_v[i_b, i_v].prec @ self.pcg_fem_state_v[i_b, i_v].r
            self.pcg_fem_state_v[i_b, i_v].p = self.pcg_fem_state_v[i_b, i_v].z
            self.pcg_state[i_b].rTr += self.pcg_fem_state_v[i_b, i_v].r.dot(self.pcg_fem_state_v[i_b, i_v].r)
            self.pcg_state[i_b].rTz += self.pcg_fem_state_v[i_b, i_v].r.dot(self.pcg_fem_state_v[i_b, i_v].z)

    @qd.func
    def compute_rigid_mass_mat_vec_product(self, vec, out, active, rigid_info: array_class.RigidInfo):
        """
        Compute the rigid mass matrix-vector product.
        """
        out.fill(0.0)
        for i_b, i_d0, i_d1 in qd.ndrange(self._B, self.rigid_solver.n_dofs, self.rigid_solver.n_dofs):
            if not active[i_b]:
                continue
            out[i_b, i_d1] += rigid_info.mass_mat[i_d1, i_d0, i_b] * vec[i_b, i_d0]

    # FIXME: This following two rigid solves are duplicated with the one in rigid_solver.py:func_solve_mass_batched
    # Consider refactoring.
    @qd.func
    def rigid_solve_pcg(self, vec, out, entities_info: array_class.EntitiesInfo, rigid_info: array_class.RigidInfo):
        # Step 1: Solve w st. L^T @ w = y
        for i_b, i_e in qd.ndrange(self._B, self.rigid_solver.n_entities):
            if not self.batch_pcg_active[i_b]:
                continue
            entity_dof_start = entities_info.dof_start[i_e]
            entity_dof_end = entities_info.dof_end[i_e]
            n_dofs = entities_info.n_dofs[i_e]
            for i_d_ in range(n_dofs):
                i_d = entity_dof_end - i_d_ - 1
                out[i_b, i_d] = vec[i_b, i_d]
                for j_d in range(i_d + 1, entity_dof_end):
                    out[i_b, i_d] -= rigid_info.mass_mat_L[j_d, i_d, i_b] * out[i_b, j_d]

        # Step 2: z = D^{-1} w
        for i_b, i_d in qd.ndrange(self._B, self.rigid_solver.n_dofs):
            if not self.batch_pcg_active[i_b]:
                continue
            out[i_b, i_d] *= rigid_info.mass_mat_D_inv[i_d, i_b]

        # Step 3: Solve x st. L @ x = z
        for i_b, i_e in qd.ndrange(self._B, self.rigid_solver.n_entities):
            if not self.batch_pcg_active[i_b]:
                continue
            entity_dof_start = entities_info.dof_start[i_e]
            entity_dof_end = entities_info.dof_end[i_e]
            n_dofs = entities_info.n_dofs[i_e]
            for i_d in range(entity_dof_start, entity_dof_end):
                for j_d in range(entity_dof_start, i_d):
                    out[i_b, i_d] -= rigid_info.mass_mat_L[i_d, j_d, i_b] * out[i_b, j_d]

    @qd.func
    def rigid_solve_jacobian(
        self,
        vec,
        out,
        n_contact_pairs,
        i_bs,
        dim,
        entities_info: array_class.EntitiesInfo,
        rigid_info: array_class.RigidInfo,
    ):
        # Step 1: Solve w st. L^T @ w = y
        for i_p, i_e, k in qd.ndrange(n_contact_pairs, self.rigid_solver.n_entities, dim):
            i_b = i_bs[i_p]
            entity_dof_start = entities_info.dof_start[i_e]
            entity_dof_end = entities_info.dof_end[i_e]
            n_dofs = entities_info.n_dofs[i_e]
            for i_d_ in range(n_dofs):
                i_d = entity_dof_end - i_d_ - 1
                out[i_p, i_d][k] = vec[i_p, i_d][k]
                for j_d in range(i_d + 1, entity_dof_end):
                    out[i_p, i_d][k] -= rigid_info.mass_mat_L[j_d, i_d, i_b] * out[i_p, j_d][k]

        # Step 2: z = D^{-1} w
        for i_p, i_d, k in qd.ndrange(n_contact_pairs, self.rigid_solver.n_dofs, dim):
            i_b = i_bs[i_p]
            out[i_p, i_d][k] *= rigid_info.mass_mat_D_inv[i_d, i_b]

        # Step 3: Solve x st. L @ x = z
        for i_p, i_e, k in qd.ndrange(n_contact_pairs, self.rigid_solver.n_entities, dim):
            i_b = i_bs[i_p]
            entity_dof_start = entities_info.dof_start[i_e]
            entity_dof_end = entities_info.dof_end[i_e]
            n_dofs = entities_info.n_dofs[i_e]
            for i_d in range(entity_dof_start, entity_dof_end):
                for j_d in range(entity_dof_start, i_d):
                    out[i_p, i_d][k] -= rigid_info.mass_mat_L[i_d, j_d, i_b] * out[i_p, j_d][k]

    @qd.func
    def init_rigid_pcg_solve(self, entities_info: array_class.EntitiesInfo, rigid_info: array_class.RigidInfo):
        for i_b, i_d in qd.ndrange(self._B, self.rigid_solver.n_dofs):
            if not self.batch_pcg_active[i_b]:
                continue
            self.pcg_rigid_state_dof[i_b, i_d].x = 0.0
            self.pcg_rigid_state_dof[i_b, i_d].r = -self.rigid_state_dof.gradient[i_b, i_d]
            self.pcg_state[i_b].rTr += self.pcg_rigid_state_dof[i_b, i_d].r ** 2

        self.rigid_solve_pcg(
            self.pcg_rigid_state_dof.r, self.pcg_rigid_state_dof.z, entities_info=entities_info, rigid_info=rigid_info
        )

        for i_b, i_d in qd.ndrange(self._B, self.rigid_solver.n_dofs):
            if not self.batch_pcg_active[i_b]:
                continue
            self.pcg_rigid_state_dof[i_b, i_d].p = self.pcg_rigid_state_dof[i_b, i_d].z
            self.pcg_state[i_b].rTz += self.pcg_rigid_state_dof[i_b, i_d].r * self.pcg_rigid_state_dof[i_b, i_d].z

    @qd.func
    def init_pcg_active(self):
        for i_b in qd.ndrange(self._B):
            if not self.batch_pcg_active[i_b]:
                continue
            self.batch_pcg_active[i_b] = self.pcg_state[i_b].rTr > self._pcg_threshold

    def one_pcg_iter(self):
        self._kernel_one_pcg_iter(
            entities_info=self.rigid_solver.dyn_info.entities, rigid_info=self.rigid_solver.rigid_info
        )

    @qd.kernel
    def _kernel_one_pcg_iter(self, entities_info: array_class.EntitiesInfo, rigid_info: array_class.RigidInfo):
        self.compute_pcg_matrix_vector_product(rigid_info=rigid_info)
        self.clear_pcg_state()
        self.compute_pcg_pTAp()
        self.compute_alpha()
        self.compute_pcg_state(entities_info=entities_info, rigid_info=rigid_info)
        self.check_pcg_convergence()
        self.compute_p()

    @qd.func
    def compute_pcg_matrix_vector_product(self, rigid_info: array_class.RigidInfo):
        """
        Compute the matrix-vector product Ap used in the Preconditioned Conjugate Gradient method.
        """
        if qd.static(self.fem_solver.is_active):
            self.compute_fem_pcg_matrix_vector_product()
        if qd.static(self.rigid_solver.is_active):
            self.compute_rigid_pcg_matrix_vector_product(rigid_info=rigid_info)
        # Constraint
        if qd.static(self.rigid_solver.is_active and self.rigid_solver.n_equalities > 0):
            self.equality_constraint_handler.compute_Ap()
        # Contact
        for contact in qd.static(self.contact_handlers):
            contact.compute_pcg_matrix_vector_product()

    @qd.func
    def clear_pcg_state(self):
        for i_b in qd.ndrange(self._B):
            if not self.batch_pcg_active[i_b]:
                continue
            self.pcg_state[i_b].pTAp = 0.0
            self.pcg_state[i_b].rTr_new = 0.0
            self.pcg_state[i_b].rTz_new = 0.0

    @qd.func
    def compute_pcg_pTAp(self):
        """
        Compute the product p^T @ A @ p used in the Preconditioned Conjugate Gradient method.

        Notes
        -----
        Reference: https://en.wikipedia.org/wiki/Conjugate_gradient_method#The_preconditioned_conjugate_gradient_method
        """
        if qd.static(self.fem_solver.is_active):
            self.compute_fem_pcg_pTAp()
        if qd.static(self.rigid_solver.is_active):
            self.compute_rigid_pcg_pTAp()

    @qd.func
    def compute_fem_pcg_pTAp(self):
        for i_b, i_v in qd.ndrange(self._B, self.fem_solver.n_vertices):
            if not self.batch_pcg_active[i_b]:
                continue
            self.pcg_state[i_b].pTAp += self.pcg_fem_state_v[i_b, i_v].p.dot(self.pcg_fem_state_v[i_b, i_v].Ap)

    @qd.func
    def compute_rigid_pcg_pTAp(self):
        for i_b, i_d in qd.ndrange(self._B, self.rigid_solver.n_dofs):
            if not self.batch_pcg_active[i_b]:
                continue
            self.pcg_state[i_b].pTAp += self.pcg_rigid_state_dof[i_b, i_d].p * self.pcg_rigid_state_dof[i_b, i_d].Ap

    @qd.func
    def compute_alpha(self):
        for i_b in qd.ndrange(self._B):
            if not self.batch_pcg_active[i_b]:
                continue
            self.pcg_state[i_b].alpha = self.pcg_state[i_b].rTz / self.pcg_state[i_b].pTAp

    @qd.func
    def compute_pcg_state(self, entities_info: array_class.EntitiesInfo, rigid_info: array_class.RigidInfo):
        if qd.static(self.fem_solver.is_active):
            self.compute_fem_pcg_state()
        if qd.static(self.rigid_solver.is_active):
            self.compute_rigid_pcg_state(entities_info=entities_info, rigid_info=rigid_info)

    @qd.func
    def compute_fem_pcg_state(self):
        for i_b, i_v in qd.ndrange(self._B, self.fem_solver.n_vertices):
            if not self.batch_pcg_active[i_b]:
                continue
            self.pcg_fem_state_v[i_b, i_v].x = (
                self.pcg_fem_state_v[i_b, i_v].x + self.pcg_state[i_b].alpha * self.pcg_fem_state_v[i_b, i_v].p
            )
            self.pcg_fem_state_v[i_b, i_v].r = (
                self.pcg_fem_state_v[i_b, i_v].r - self.pcg_state[i_b].alpha * self.pcg_fem_state_v[i_b, i_v].Ap
            )
            self.pcg_fem_state_v[i_b, i_v].z = self.pcg_fem_state_v[i_b, i_v].prec @ self.pcg_fem_state_v[i_b, i_v].r
            self.pcg_state[i_b].rTr_new += self.pcg_fem_state_v[i_b, i_v].r.norm_sqr()
            self.pcg_state[i_b].rTz_new += self.pcg_fem_state_v[i_b, i_v].r.dot(self.pcg_fem_state_v[i_b, i_v].z)

    @qd.func
    def compute_rigid_pcg_state(self, entities_info: array_class.EntitiesInfo, rigid_info: array_class.RigidInfo):
        for i_b, i_d in qd.ndrange(self._B, self.rigid_solver.n_dofs):
            if not self.batch_pcg_active[i_b]:
                continue
            self.pcg_rigid_state_dof[i_b, i_d].x = (
                self.pcg_rigid_state_dof[i_b, i_d].x + self.pcg_state[i_b].alpha * self.pcg_rigid_state_dof[i_b, i_d].p
            )
            self.pcg_rigid_state_dof[i_b, i_d].r = (
                self.pcg_rigid_state_dof[i_b, i_d].r - self.pcg_state[i_b].alpha * self.pcg_rigid_state_dof[i_b, i_d].Ap
            )
            self.pcg_state[i_b].rTr_new += self.pcg_rigid_state_dof[i_b, i_d].r * self.pcg_rigid_state_dof[i_b, i_d].r

        self.rigid_solve_pcg(
            self.pcg_rigid_state_dof.r, self.pcg_rigid_state_dof.z, entities_info=entities_info, rigid_info=rigid_info
        )

        for i_b, i_d in qd.ndrange(self._B, self.rigid_solver.n_dofs):
            if not self.batch_pcg_active[i_b]:
                continue
            self.pcg_state[i_b].rTz_new += self.pcg_rigid_state_dof[i_b, i_d].r * self.pcg_rigid_state_dof[i_b, i_d].z

    @qd.func
    def check_pcg_convergence(self):
        # check convergence
        for i_b in qd.ndrange(self._B):
            if not self.batch_pcg_active[i_b]:
                continue
            self.batch_pcg_active[i_b] = self.pcg_state[i_b].rTr_new > self._pcg_threshold
        # update beta, rTr, rTz
        for i_b in qd.ndrange(self._B):
            if not self.batch_pcg_active[i_b]:
                continue
            self.pcg_state[i_b].beta = self.pcg_state[i_b].rTz_new / self.pcg_state[i_b].rTz
            self.pcg_state[i_b].rTr = self.pcg_state[i_b].rTr_new
            self.pcg_state[i_b].rTz = self.pcg_state[i_b].rTz_new

    @qd.func
    def compute_p(self):
        if qd.static(self.fem_solver.is_active):
            self.compute_fem_p()
        if qd.static(self.rigid_solver.is_active):
            self.compute_rigid_p()

    @qd.func
    def compute_fem_p(self):
        for i_b, i_v in qd.ndrange(self._B, self.fem_solver.n_vertices):
            if not self.batch_pcg_active[i_b]:
                continue
            self.pcg_fem_state_v[i_b, i_v].p = (
                self.pcg_fem_state_v[i_b, i_v].z + self.pcg_state[i_b].beta * self.pcg_fem_state_v[i_b, i_v].p
            )

    @qd.func
    def compute_rigid_p(self):
        for i_b, i_d in qd.ndrange(self._B, self.rigid_solver.n_dofs):
            if not self.batch_pcg_active[i_b]:
                continue
            self.pcg_rigid_state_dof[i_b, i_d].p = (
                self.pcg_rigid_state_dof[i_b, i_d].z + self.pcg_state[i_b].beta * self.pcg_rigid_state_dof[i_b, i_d].p
            )

    def pcg_solve(self):
        self.init_pcg_solve(entities_info=self.rigid_solver.dyn_info.entities, rigid_info=self.rigid_solver.rigid_info)
        for i in range(self._n_pcg_iterations):
            self.one_pcg_iter()

    @qd.func
    def compute_total_energy(
        self,
        i_step: qd.i32,
        energy: qd.template(),
        dofs_state: array_class.DofsState,
        rigid_info: array_class.RigidInfo,
    ):
        energy.fill(0.0)
        if qd.static(self.fem_solver.is_active):
            self.compute_fem_energy(i_step, energy)
        if qd.static(self.rigid_solver.is_active):
            self.compute_rigid_energy(energy, dofs_state=dofs_state, rigid_info=rigid_info)
        # Constraint
        if qd.static(self.rigid_solver.is_active and self.rigid_solver.n_equalities > 0):
            self.equality_constraint_handler.compute_energy(energy)
        # Contact
        for contact in qd.static(self.contact_handlers):
            contact.compute_energy(energy)

    @qd.func
    def compute_fem_energy(self, i_step: qd.i32, energy: qd.template()):
        dt2 = self.fem_solver._substep_dt**2
        damping_alpha_factor = self.fem_solver._damping_alpha * self.fem_solver._substep_dt + 1.0
        damping_beta_factor = self.fem_solver._damping_beta / self.fem_solver._substep_dt + 1.0

        # Inertia
        for i_b, i_v in qd.ndrange(self._B, self.fem_solver.n_vertices):
            if not self.batch_linesearch_active[i_b]:
                continue
            self.fem_state_v.v_diff[i_b, i_v] = (
                self.fem_state_v.v[i_b, i_v] - self.fem_solver.elements_v[i_step + 1, i_v, i_b].vel
            )
            energy[i_b] += (
                0.5
                * self.fem_solver.elements_v_info[i_v].mass_over_dt2
                * self.fem_state_v.v_diff[i_b, i_v].norm_sqr()
                * dt2
                * damping_alpha_factor
            )

        # Elastic
        for i_b, i_e in qd.ndrange(self._B, self.fem_solver.n_elements):
            if not self.batch_linesearch_active[i_b]:
                continue

            V_dt2 = self.fem_solver.elements_i[i_e].V * dt2
            B = self.fem_solver.elements_i[i_e].B
            S = qd.Matrix.zero(gs.qd_float, 4, 3)
            S[:3, :] = B
            S[3, :] = -B[0, :] - B[1, :] - B[2, :]
            i_vs = self.fem_solver.elements_i[i_e].el2v

            if qd.static(self.fem_solver._enable_vertex_constraints):
                for i in qd.static(range(4)):
                    if self.fem_solver.vertex_constraints.is_constrained[i_vs[i], i_b]:
                        S[i, :] = qd.Vector.zero(gs.qd_float, 3)

            p9, H9_p9 = self.compute_elastic_products(i_b, i_e, S, i_vs, self.fem_state_v.v_diff)
            energy[i_b] += 0.5 * p9.dot(H9_p9) * damping_beta_factor * V_dt2

    @qd.func
    def compute_rigid_energy(
        self, energy: qd.template(), dofs_state: array_class.DofsState, rigid_info: array_class.RigidInfo
    ):
        # Kinetic energy
        for i_b, i_d in qd.ndrange(self._B, self.rigid_solver.n_dofs):
            if not self.batch_linesearch_active[i_b]:
                continue
            self.rigid_state_dof.v_diff[i_b, i_d] = self.rigid_state_dof.v[i_b, i_d] - dofs_state.vel[i_d, i_b]
        self.compute_rigid_mass_mat_vec_product(
            self.rigid_state_dof.v_diff,
            self.rigid_state_dof.mass_v_diff,
            self.batch_linesearch_active,
            rigid_info=rigid_info,
        )
        for i_b, i_d in qd.ndrange(self._B, self.rigid_solver.n_dofs):
            if not self.batch_linesearch_active[i_b]:
                continue
            energy[i_b] += 0.5 * self.rigid_state_dof.v_diff[i_b, i_d] * self.rigid_state_dof.mass_v_diff[i_b, i_d]

    @qd.kernel
    def init_exact_linesearch(
        self, i_step: qd.i32, dofs_state: array_class.DofsState, rigid_info: array_class.RigidInfo
    ):
        self._func_init_linesearch(self._linesearch_max_step_size)
        self.compute_total_energy(
            i_step, self.linesearch_state.prev_energy, dofs_state=dofs_state, rigid_info=rigid_info
        )
        self.prepare_search_direction_data(rigid_info=rigid_info)
        self.update_velocity_linesearch()
        self.compute_line_energy_gradient_hessian(i_step, dofs_state=dofs_state)
        self.check_initial_exact_linesearch_convergence()
        self.init_newton_linesearch()

    @qd.func
    def init_newton_linesearch(self):
        for i_b in qd.ndrange(self._B):
            if not self.batch_linesearch_active[i_b]:
                continue
            self.linesearch_state[i_b].dell_scale = -self.linesearch_state[i_b].m
            self.linesearch_state[i_b].step_size = qd.min(
                -self.linesearch_state[i_b].m / self.linesearch_state[i_b].d2ell_dalpha2, self._linesearch_max_step_size
            )
            self.linesearch_state[i_b].alpha_min = 0.0
            self.linesearch_state[i_b].alpha_max = self._linesearch_max_step_size
            self.linesearch_state[i_b].f_lower = -1.0
            self.linesearch_state[i_b].f_upper = (
                self.linesearch_state[i_b].dell_dalpha / self.linesearch_state[i_b].dell_scale
            )
            self.linesearch_state[i_b].alpha_tol = self._linesearch_ftol * self.linesearch_state[i_b].step_size
            self.linesearch_state[i_b].minus_dalpha = (
                self.linesearch_state[i_b].alpha_min - self.linesearch_state[i_b].alpha_max
            )
            self.linesearch_state[i_b].minus_dalpha_prev = self.linesearch_state[i_b].minus_dalpha
            if qd.abs(self.linesearch_state[i_b].f_lower) < self._linesearch_ftol:
                self.batch_linesearch_active[i_b] = False
                self.linesearch_state[i_b].step_size = self.linesearch_state[i_b].alpha_min
            if qd.abs(self.linesearch_state[i_b].f_upper) < self._linesearch_ftol:
                self.batch_linesearch_active[i_b] = False
                self.linesearch_state[i_b].step_size = self.linesearch_state[i_b].alpha_max

    @qd.func
    def compute_line_energy_gradient_hessian(self, i_step: qd.i32, dofs_state: array_class.DofsState):
        self.init_linesearch_energy_gradient_hessian()
        if qd.static(self.fem_solver.is_active):
            self.compute_fem_energy_alpha(i_step, self.linesearch_state.energy)
            self.compute_fem_gradient_alpha(i_step)

        if qd.static(self.rigid_solver.is_active):
            self.compute_rigid_energy_alpha(self.linesearch_state.energy, dofs_state=dofs_state)
            self.compute_rigid_gradient_alpha(dofs_state=dofs_state)
        # Constraint
        if qd.static(self.rigid_solver.is_active and self.rigid_solver.n_equalities > 0):
            self.equality_constraint_handler.compute_energy_gamma_G()
            self.equality_constraint_handler.update_gradient_hessian_alpha()
        # Contact
        for contact in qd.static(self.contact_handlers):
            contact.compute_energy_gamma_G()
            contact.update_gradient_hessian_alpha()

    @qd.func
    def init_linesearch_energy_gradient_hessian(self):
        energy = qd.static(self.linesearch_state.energy)
        alpha = qd.static(self.linesearch_state.step_size)
        for i_b in qd.ndrange(self._B):
            if not self.batch_linesearch_active[i_b]:
                continue

            # energy
            energy[i_b] = (
                self.linesearch_state.prev_energy[i_b]
                + 0.5 * alpha[i_b] ** 2 * self.linesearch_state[i_b].d2ellA_dalpha2
            )

            # gradient
            self.linesearch_state[i_b].dell_dalpha = 0.0

            # hessian
            self.linesearch_state.d2ell_dalpha2[i_b] = self.linesearch_state.d2ellA_dalpha2[i_b]

    @qd.func
    def compute_fem_gradient_alpha(self, i_step: qd.i32):
        dp = qd.static(self.linesearch_fem_state_v.dp)
        v = qd.static(self.fem_state_v.v)
        v_star = qd.static(self.fem_solver.elements_v.vel)
        for i_b, i_v in qd.ndrange(self._B, self.fem_solver.n_vertices):
            if not self.batch_linesearch_active[i_b]:
                continue
            self.linesearch_state.dell_dalpha[i_b] += dp[i_b, i_v].dot(v[i_b, i_v] - v_star[i_step + 1, i_v, i_b])

    @qd.func
    def compute_rigid_gradient_alpha(self, dofs_state: array_class.DofsState):
        dp = qd.static(self.linesearch_rigid_state_dof.dp)
        v = qd.static(self.rigid_state_dof.v)
        v_star = qd.static(dofs_state.vel)
        for i_b, i_d in qd.ndrange(self._B, self.rigid_solver.n_dofs):
            if not self.batch_linesearch_active[i_b]:
                continue
            self.linesearch_state.dell_dalpha[i_b] += dp[i_b, i_d] * (v[i_b, i_d] - v_star[i_d, i_b])

    @qd.func
    def compute_fem_energy_alpha(self, i_step: qd.i32, energy: qd.template()):
        alpha = qd.static(self.linesearch_state.step_size)
        dp = qd.static(self.linesearch_fem_state_v.dp)
        v = qd.static(self.fem_state_v.v)
        v_star = qd.static(self.fem_solver.elements_v.vel)
        for i_b, i_v in qd.ndrange(self._B, self.fem_solver.n_vertices):
            if not self.batch_linesearch_active[i_b]:
                continue
            energy[i_b] += alpha[i_b] * dp[i_b, i_v].dot(v[i_b, i_v] - v_star[i_step + 1, i_v, i_b])

    @qd.func
    def compute_rigid_energy_alpha(self, energy: qd.template(), dofs_state: array_class.DofsState):
        alpha = qd.static(self.linesearch_state.step_size)
        dp = qd.static(self.linesearch_rigid_state_dof.dp)
        v = qd.static(self.rigid_state_dof.v)
        v_star = qd.static(dofs_state.vel)
        for i_b, i_d in qd.ndrange(self._B, self.rigid_solver.n_dofs):
            if not self.batch_linesearch_active[i_b]:
                continue
            energy[i_b] += alpha[i_b] * dp[i_b, i_d] * (v[i_b, i_d] - v_star[i_d, i_b])

    @qd.func
    def prepare_search_direction_data(self, rigid_info: array_class.RigidInfo):
        if qd.static(self.fem_solver.is_active):
            self.prepare_fem_search_direction_data()
        if qd.static(self.rigid_solver.is_active):
            self.prepare_rigid_search_direction_data(rigid_info=rigid_info)
        # Constraint
        if qd.static(self.rigid_solver.is_active and self.rigid_solver.n_equalities > 0):
            self.equality_constraint_handler.prepare_search_direction_data()
        # Contact
        for contact in qd.static(self.contact_handlers):
            contact.prepare_search_direction_data()
        self.compute_d2ellA_dalpha2()

    @qd.func
    def compute_d2ellA_dalpha2(self):
        for i_b in qd.ndrange(self._B):
            self.linesearch_state[i_b].d2ellA_dalpha2 = 0.0
        if qd.static(self.fem_solver.is_active):
            self.compute_fem_d2ellA_dalpha2()
        if qd.static(self.rigid_solver.is_active):
            self.compute_rigid_d2ellA_dalpha2()

    @qd.func
    def compute_fem_d2ellA_dalpha2(self):
        for i_b, i_v in qd.ndrange(self._B, self.fem_solver.n_vertices):
            if not self.batch_linesearch_active[i_b]:
                continue
            self.linesearch_state[i_b].d2ellA_dalpha2 += self.pcg_fem_state_v[i_b, i_v].x.dot(
                self.linesearch_fem_state_v[i_b, i_v].dp
            )

    @qd.func
    def compute_rigid_d2ellA_dalpha2(self):
        for i_b, i_d in qd.ndrange(self._B, self.rigid_solver.n_dofs):
            if not self.batch_linesearch_active[i_b]:
                continue
            self.linesearch_state[i_b].d2ellA_dalpha2 += (
                self.pcg_rigid_state_dof[i_b, i_d].x * self.linesearch_rigid_state_dof[i_b, i_d].dp
            )

    @qd.func
    def prepare_fem_search_direction_data(self):
        self.compute_fem_matrix_vector_product(
            self.pcg_fem_state_v.x, self.linesearch_fem_state_v.dp, self.batch_linesearch_active
        )

    @qd.func
    def prepare_rigid_search_direction_data(self, rigid_info: array_class.RigidInfo):
        self.compute_rigid_mass_mat_vec_product(
            self.pcg_rigid_state_dof.x,
            self.linesearch_rigid_state_dof.dp,
            self.batch_linesearch_active,
            rigid_info=rigid_info,
        )

    @qd.func
    def _func_init_linesearch(self, step_size: float):
        for i_b in qd.ndrange(self._B):
            self.batch_linesearch_active[i_b] = self.batch_active[i_b]
            if not self.batch_linesearch_active[i_b]:
                continue
            self.linesearch_state[i_b].step_size = step_size
            self.linesearch_state[i_b].m = 0.0

        if qd.static(self.fem_solver.is_active):
            self._func_init_fem_linesearch()
        if qd.static(self.rigid_solver.is_active):
            self._func_init_rigid_linesearch()

    @qd.func
    def _func_init_fem_linesearch(self):
        for i_b, i_v in qd.ndrange(self._B, self.fem_solver.n_vertices):
            if not self.batch_linesearch_active[i_b]:
                continue
            self.linesearch_state[i_b].m += self.pcg_fem_state_v[i_b, i_v].x.dot(self.fem_state_v.gradient[i_b, i_v])
            self.linesearch_fem_state_v[i_b, i_v].x_prev = self.fem_state_v.v[i_b, i_v]

    @qd.func
    def _func_init_rigid_linesearch(self):
        for i_b, i_d in qd.ndrange(self._B, self.rigid_solver.n_dofs):
            if not self.batch_linesearch_active[i_b]:
                continue
            self.linesearch_state[i_b].m += (
                self.pcg_rigid_state_dof[i_b, i_d].x * self.rigid_state_dof.gradient[i_b, i_d]
            )
            self.linesearch_rigid_state_dof[i_b, i_d].x_prev = self.rigid_state_dof.v[i_b, i_d]

    @qd.func
    def check_initial_exact_linesearch_convergence(self):
        for i_b in qd.ndrange(self._B):
            if not self.batch_linesearch_active[i_b]:
                continue
            self.batch_linesearch_active[i_b] = self.linesearch_state[i_b].dell_dalpha > 0.0

        if qd.static(self.fem_solver.is_active):
            self.update_initial_fem_state()
        if qd.static(self.rigid_solver.is_active):
            self.update_initial_rigid_state()

        # When tolerance is small but gradient norm is small, take step 1.0 and end, this is a rare case, directly
        # copied from drake
        # Link: https://github.com/RobotLocomotion/drake/blob/3bb00e611983fb894151c547776d5aa85abe9139/multibody/contact_solvers/sap/sap_solver.cc#L625
        for i_b in range(self._B):
            if not self.batch_linesearch_active[i_b]:
                continue
            err_threshold = (
                self._sap_convergence_atol + self._sap_convergence_rtol * self.linesearch_state[i_b].prev_energy
            )
            if -self.linesearch_state[i_b].m < err_threshold:
                self.batch_linesearch_active[i_b] = False
                self.linesearch_state[i_b].step_size = 1.0

    @qd.func
    def update_initial_fem_state(self):
        for i_b, i_v in qd.ndrange(self._B, self.fem_solver.n_vertices):
            if not self.batch_linesearch_active[i_b]:
                continue
            err_threshold = (
                self._sap_convergence_atol + self._sap_convergence_rtol * self.linesearch_state[i_b].prev_energy
            )
            if -self.linesearch_state[i_b].m < err_threshold:
                self.fem_state_v.v[i_b, i_v] = (
                    self.linesearch_fem_state_v[i_b, i_v].x_prev + self.pcg_fem_state_v[i_b, i_v].x
                )

    @qd.func
    def update_initial_rigid_state(self):
        for i_b, i_d in qd.ndrange(self._B, self.rigid_solver.n_dofs):
            if not self.batch_linesearch_active[i_b]:
                continue
            err_threshold = (
                self._sap_convergence_atol + self._sap_convergence_rtol * self.linesearch_state[i_b].prev_energy
            )
            if -self.linesearch_state[i_b].m < err_threshold:
                self.rigid_state_dof.v[i_b, i_d] = (
                    self.linesearch_rigid_state_dof[i_b, i_d].x_prev + self.pcg_rigid_state_dof[i_b, i_d].x
                )

    def one_linesearch_iter(self, i_step: qd.i32):
        self.update_velocity_linesearch()
        self.compute_total_energy(i_step, self.linesearch_state.energy)
        self.check_linesearch_convergence()

    @qd.func
    def update_velocity_linesearch(self):
        if qd.static(self.fem_solver.is_active):
            self.update_fem_velocity_linesearch()
        if qd.static(self.rigid_solver.is_active):
            self.update_rigid_velocity_linesearch()

    @qd.func
    def update_fem_velocity_linesearch(self):
        for i_b, i_v in qd.ndrange(self._B, self.fem_solver.n_vertices):
            if not self.batch_linesearch_active[i_b]:
                continue
            self.fem_state_v.v[i_b, i_v] = (
                self.linesearch_fem_state_v[i_b, i_v].x_prev
                + self.linesearch_state[i_b].step_size * self.pcg_fem_state_v[i_b, i_v].x
            )

    @qd.func
    def update_rigid_velocity_linesearch(self):
        for i_b, i_d in qd.ndrange(self._B, self.rigid_solver.n_dofs):
            if not self.batch_linesearch_active[i_b]:
                continue
            self.rigid_state_dof.v[i_b, i_d] = (
                self.linesearch_rigid_state_dof[i_b, i_d].x_prev
                + self.linesearch_state[i_b].step_size * self.pcg_rigid_state_dof[i_b, i_d].x
            )

    def exact_linesearch(self, i_step: qd.i32):
        """
        Note
        ------
        Exact line search using rtsafe
        https://github.com/RobotLocomotion/drake/blob/master/multibody/contact_solvers/sap/sap_solver.h#L393
        """
        self.init_exact_linesearch(
            i_step, dofs_state=self.rigid_solver.dyn_state.dofs, rigid_info=self.rigid_solver.rigid_info
        )
        for i in range(self._n_linesearch_iterations):
            self.one_exact_linesearch_iter(i_step, dofs_state=self.rigid_solver.dyn_state.dofs)

    @qd.kernel
    def one_exact_linesearch_iter(self, i_step: qd.i32, dofs_state: array_class.DofsState):
        self.update_velocity_linesearch()
        self.compute_line_energy_gradient_hessian(i_step, dofs_state=dofs_state)
        self.compute_f_df_bracket()
        self.find_next_step_size()

    @qd.func
    def compute_f_df_bracket(self):
        """
        Compute the function (derivative of total energy) value and its derivative to alpha.
        Update the bracket for the next step size.

        The bracket is defined by [alpha_min, alpha_max] which is the range that contains the root of df/dalpha = 0.
        """
        for i_b in qd.ndrange(self._B):
            if not self.batch_linesearch_active[i_b]:
                continue
            self.linesearch_state[i_b].f = (
                self.linesearch_state[i_b].dell_dalpha / self.linesearch_state[i_b].dell_scale
            )
            self.linesearch_state[i_b].df = (
                self.linesearch_state[i_b].d2ell_dalpha2 / self.linesearch_state[i_b].dell_scale
            )
            if qd.math.sign(self.linesearch_state[i_b].f) != qd.math.sign(self.linesearch_state[i_b].f_upper):
                self.linesearch_state[i_b].alpha_min = self.linesearch_state[i_b].step_size
                self.linesearch_state[i_b].f_lower = self.linesearch_state[i_b].f
            else:
                self.linesearch_state[i_b].alpha_max = self.linesearch_state[i_b].step_size
                self.linesearch_state[i_b].f_upper = self.linesearch_state[i_b].f
            if qd.abs(self.linesearch_state[i_b].f) < self._linesearch_ftol:
                self.batch_linesearch_active[i_b] = False

    @qd.func
    def find_next_step_size(self):
        for i_b in qd.ndrange(self._B):
            if not self.batch_linesearch_active[i_b]:
                continue
            newton_is_slow = 2.0 * qd.abs(self.linesearch_state[i_b].f) > qd.abs(
                self.linesearch_state[i_b].minus_dalpha_prev * self.linesearch_state[i_b].df
            )
            self.linesearch_state[i_b].minus_dalpha_prev = self.linesearch_state[i_b].minus_dalpha
            if newton_is_slow:
                # bisect
                self.linesearch_state[i_b].minus_dalpha = 0.5 * (
                    self.linesearch_state[i_b].alpha_min - self.linesearch_state[i_b].alpha_max
                )
                self.linesearch_state[i_b].step_size = (
                    self.linesearch_state[i_b].alpha_min - self.linesearch_state[i_b].minus_dalpha
                )
            else:
                # newton
                self.linesearch_state[i_b].minus_dalpha = self.linesearch_state[i_b].f / self.linesearch_state[i_b].df
                self.linesearch_state[i_b].step_size = (
                    self.linesearch_state[i_b].step_size - self.linesearch_state[i_b].minus_dalpha
                )
                if (
                    self.linesearch_state[i_b].step_size <= self.linesearch_state[i_b].alpha_min
                    or self.linesearch_state[i_b].step_size >= self.linesearch_state[i_b].alpha_max
                ):
                    # bisect
                    self.linesearch_state[i_b].minus_dalpha = 0.5 * (
                        self.linesearch_state[i_b].alpha_min - self.linesearch_state[i_b].alpha_max
                    )
                    self.linesearch_state[i_b].step_size = (
                        self.linesearch_state[i_b].alpha_min - self.linesearch_state[i_b].minus_dalpha
                    )
            if qd.abs(self.linesearch_state[i_b].minus_dalpha) < self.linesearch_state[i_b].alpha_tol:
                self.batch_linesearch_active[i_b] = False

    # ------------------------------------------------------------------------------------
    # ----------------------------------- Properties -------------------------------------
    # ------------------------------------------------------------------------------------
    @property
    def active_solvers(self):
        """All the active solvers managed by the scene's simulator."""
        return self.sim.active_solvers


@qd.data_oriented
class BaseConstraintHandler(RBC):
    """
    Base class for constraint handling in SAPCoupler.
    """

    def __init__(self, simulator: "Simulator", stiffness: float = 1e8, beta: float = 0.1) -> None:
        self.sim = simulator
        self.stiffness = stiffness
        self.beta = beta
        self._B = simulator._B
        self.coupler = simulator.coupler
        self.sap_constraint_info_type = qd.types.struct(
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
    def compute_constraint_regularization(self, sap_info, i_c, w_rms, time_step):
        beta_factor = self.beta**2 / (4.0 * qd.math.pi**2)
        dt2 = time_step**2
        k = sap_info[i_c].k
        R = max(beta_factor * w_rms, 1.0 / (dt2 * k))
        sap_info[i_c].R = R
        sap_info[i_c].R_inv = 1.0 / R

    @qd.func
    def compute_constraint_gamma_G(self, sap_info, i_c, vc):
        y = (sap_info[i_c].v_hat - vc) * sap_info[i_c].R_inv
        sap_info[i_c].gamma = y
        sap_info[i_c].G = sap_info[i_c].R_inv

    @qd.func
    def compute_energy(self, energy: qd.template()):
        constraints = qd.static(self.constraints)
        sap_info = qd.static(constraints.sap_info)
        for i_c in range(self.n_constraints[None]):
            i_b = constraints[i_c].batch_idx
            if self.coupler.batch_linesearch_active[i_b]:
                vc = self.compute_vc(i_c)
                self.compute_constraint_energy(sap_info, i_c, vc)
                energy[i_b] += sap_info[i_c].energy

    @qd.func
    def compute_constraint_energy(self, sap_info, i_c, vc):
        y = (sap_info[i_c].v_hat - vc) * sap_info[i_c].R_inv
        sap_info[i_c].energy = 0.5 * y**2 * sap_info[i_c].R


@qd.data_oriented
class RigidConstraintHandler(BaseConstraintHandler):
    """
    Rigid body constraints in SAPCoupler. Currently only support joint equality constraints.
    """

    def __init__(self, simulator: "Simulator", stiffness: float = 1e8, beta: float = 0.1) -> None:
        super().__init__(simulator, stiffness, beta)
        self.rigid_solver = simulator.rigid_solver
        self.constraint_solver = simulator.rigid_solver.constraint_solver
        self.max_constraints = simulator.rigid_solver.n_equalities * self._B
        self.n_constraints = qd.field(gs.qd_int, shape=())
        self.constraint_type = qd.types.struct(
            batch_idx=gs.qd_int,  # batch index
            i_dof1=gs.qd_int,  # index of the first DOF in the constraint
            i_dof2=gs.qd_int,  # index of the second DOF in the constraint
            sap_info=self.sap_constraint_info_type,  # SAP info for the constraint
        )
        self.constraints = self.constraint_type.field(shape=(self.max_constraints,))
        self.Jt = qd.field(gs.qd_float, shape=(self.max_constraints, self.rigid_solver.n_dofs))
        self.M_inv_Jt = qd.field(gs.qd_float, shape=(self.max_constraints, self.rigid_solver.n_dofs))
        self.W = qd.field(gs.qd_float, shape=(self.max_constraints,))

    @qd.kernel
    def build_constraints(
        self,
        equalities_info: array_class.EqualitiesInfo,
        joints_info: array_class.JointsInfo,
        rigid_config: qd.template(),
    ):
        self.n_constraints[None] = 0
        self.Jt.fill(0.0)
        # TODO: Maybe support different constraints for each batch in the future.
        # For now all batches have the same constraints.
        dt2 = self.sim._substep_dt**2
        for i_b, i_e in qd.ndrange(self._B, self.rigid_solver.n_equalities):
            if equalities_info.eq_type[i_e, i_b] == gs.EQUALITY_TYPE.JOINT:
                i_c = qd.atomic_add(self.n_constraints[None], 1)
                self.constraints[i_c].batch_idx = i_b
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
                i_dof2 = joints_info.dof_start[I_joint2]
                self.constraints[i_c].i_dof1 = i_dof1
                self.constraints[i_c].i_dof2 = i_dof2
                self.constraints[i_c].sap_info.k = self.stiffness
                self.constraints[i_c].sap_info.R_inv = dt2 * self.stiffness
                self.constraints[i_c].sap_info.R = 1.0 / self.constraints[i_c].sap_info.R_inv
                self.constraints[i_c].sap_info.v_hat = 0.0
                self.Jt[i_c, i_dof1] = 1.0
                self.Jt[i_c, i_dof2] = -1.0

    @qd.func
    def compute_regularization(self, dofs_state: array_class.DofsState):
        dt_inv = 1.0 / self.sim._substep_dt
        q = qd.static(dofs_state.pos)
        sap_info = qd.static(self.constraints.sap_info)
        for i_c in range(self.n_constraints[None]):
            i_b = self.constraints[i_c].batch_idx
            g0 = q[self.constraints[i_c].i_dof1, i_b] - q[self.constraints[i_c].i_dof2, i_b]
            self.constraints[i_c].sap_info.v_hat = -g0 * dt_inv
            W = self.compute_delassus(i_c)
            self.compute_constraint_regularization(sap_info, i_c, W, self.sim._substep_dt)

    @qd.func
    def compute_delassus_world_frame(self, entities_info: array_class.EntitiesInfo, rigid_info: array_class.RigidInfo):
        self.coupler.rigid_solve_jacobian(
            self.Jt,
            self.M_inv_Jt,
            self.n_constraints[None],
            self.constraints.batch_idx,
            1,
            entities_info=entities_info,
            rigid_info=rigid_info,
        )
        self.W.fill(0.0)
        for i_c, i_d in qd.ndrange(self.n_constraints[None], self.rigid_solver.n_dofs):
            self.W[i_c] += self.M_inv_Jt[i_c, i_d] * self.Jt[i_c, i_d]

    @qd.func
    def compute_delassus(self, i_c):
        return self.W[i_c]

    @qd.func
    def compute_Jx(self, i_c, x):
        i_b = self.constraints[i_c].batch_idx
        i_dof1 = self.constraints[i_c].i_dof1
        i_dof2 = self.constraints[i_c].i_dof2
        return x[i_b, i_dof1] - x[i_b, i_dof2]

    @qd.func
    def add_Jt_x(self, y, i_c, x):
        i_b = self.constraints[i_c].batch_idx
        i_dof1 = self.constraints[i_c].i_dof1
        i_dof2 = self.constraints[i_c].i_dof2
        y[i_b, i_dof1] += x
        y[i_b, i_dof2] -= x

    @qd.func
    def compute_vc(self, i_c):
        return self.compute_Jx(i_c, self.coupler.rigid_state_dof.v)

    @qd.func
    def compute_gradient_hessian_diag(self):
        constraints = qd.static(self.constraints)
        sap_info = qd.static(constraints.sap_info)
        for i_c in range(self.n_constraints[None]):
            vc = self.compute_vc(i_c)
            self.compute_constraint_gamma_G(sap_info, i_c, vc)
            self.add_Jt_x(self.coupler.rigid_state_dof.gradient, i_c, -sap_info[i_c].gamma)
            self.add_Jt_x(self.coupler.rigid_state_dof.impulse, i_c, sap_info[i_c].gamma)

    @qd.func
    def compute_Ap(self):
        constraints = qd.static(self.constraints)
        sap_info = qd.static(constraints.sap_info)
        for i_c in range(self.n_constraints[None]):
            # Jt @ G @ J @ p
            x = self.compute_Jx(i_c, self.coupler.pcg_rigid_state_dof.p)
            x = sap_info[i_c].G * x
            self.add_Jt_x(self.coupler.pcg_rigid_state_dof.Ap, i_c, x)

    @qd.func
    def prepare_search_direction_data(self):
        constraints = qd.static(self.constraints)
        sap_info = qd.static(constraints.sap_info)
        for i_c in range(self.n_constraints[None]):
            i_b = constraints[i_c].batch_idx
            if self.coupler.batch_linesearch_active[i_b]:
                sap_info[i_c].dvc = self.compute_Jx(i_c, self.coupler.pcg_rigid_state_dof.x)

    @qd.func
    def compute_energy_gamma_G(self):
        constraints = qd.static(self.constraints)
        sap_info = qd.static(constraints.sap_info)
        for i_c in range(self.n_constraints[None]):
            vc = self.compute_vc(i_c)
            self.compute_constraint_energy_gamma_G(sap_info, i_c, vc)

    @qd.func
    def compute_constraint_energy_gamma_G(self, sap_info, i_c, vc):
        self.compute_constraint_gamma_G(sap_info, i_c, vc)
        sap_info[i_c].energy = 0.5 * sap_info[i_c].gamma ** 2 * sap_info[i_c].R

    @qd.func
    def update_gradient_hessian_alpha(self):
        dvc = qd.static(self.constraints.sap_info.dvc)
        gamma = qd.static(self.constraints.sap_info.gamma)
        G = qd.static(self.constraints.sap_info.G)
        for i_c in qd.ndrange(self.n_constraints[None]):
            i_b = self.constraints[i_c].batch_idx
            if self.coupler.batch_linesearch_active[i_b]:
                self.coupler.linesearch_state.dell_dalpha[i_b] -= dvc[i_c] * gamma[i_c]
                self.coupler.linesearch_state.d2ell_dalpha2[i_b] += dvc[i_c] ** 2 * G[i_c]


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

    def __init__(self, simulator: "Simulator") -> None:
        self.sim = simulator
        self.coupler = simulator.coupler
        self.n_contact_pairs = qd.field(gs.qd_int, shape=())
        self.sap_contact_info_type = qd.types.struct(
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
        self, links_info: array_class.LinksInfo, dofs_state: array_class.DofsState, links_state: array_class.LinksState
    ):
        pass

    @qd.func
    def update_gradient_hessian_alpha(self):
        dvc = qd.static(self.contact_pairs.sap_info.dvc)
        gamma = qd.static(self.contact_pairs.sap_info.gamma)
        G = qd.static(self.contact_pairs.sap_info.G)
        for i_p in qd.ndrange(self.n_contact_pairs[None]):
            i_b = self.contact_pairs[i_p].batch_idx
            if self.coupler.batch_linesearch_active[i_b]:
                self.coupler.linesearch_state.dell_dalpha[i_b] -= dvc[i_p].dot(gamma[i_p])
                self.coupler.linesearch_state.d2ell_dalpha2[i_b] += dvc[i_p].dot(G[i_p] @ dvc[i_p])

    @qd.func
    def compute_delassus_world_frame(self, entities_info: array_class.EntitiesInfo, rigid_info: array_class.RigidInfo):
        pass

    @qd.func
    def compute_regularization(self, entities_info: array_class.EntitiesInfo, rigid_info: array_class.RigidInfo):
        self.compute_delassus_world_frame(entities_info=entities_info, rigid_info=rigid_info)
        for i_p in range(self.n_contact_pairs[None]):
            W = self.compute_delassus(i_p)
            w_rms = W.norm() / 3.0
            self.compute_contact_regularization(self.contact_pairs.sap_info, i_p, w_rms, self.sim._substep_dt)

    @qd.func
    def compute_energy_gamma_G(self):
        for i_p in range(self.n_contact_pairs[None]):
            vc = self.compute_contact_velocity(i_p)
            self.compute_contact_energy_gamma_G(self.contact_pairs.sap_info, i_p, vc)

    @qd.func
    def compute_energy(self, energy: qd.template()):
        sap_info = qd.static(self.contact_pairs.sap_info)
        for i_p in range(self.n_contact_pairs[None]):
            i_b = self.contact_pairs[i_p].batch_idx
            if self.coupler.batch_linesearch_active[i_b]:
                vc = self.compute_contact_velocity(i_p)
                self.compute_contact_energy(sap_info, i_p, vc)
                energy[i_b] += sap_info[i_p].energy

    @qd.func
    def compute_contact_gamma_G(self, sap_info, i_p, vc):
        y = qd.Vector([0.0, 0.0, sap_info[i_p].vn_hat]) - vc
        y[0] *= sap_info[i_p].Rt_inv
        y[1] *= sap_info[i_p].Rt_inv
        y[2] *= sap_info[i_p].Rn_inv
        yr = y[:2].norm(gs.EPS)
        yn = y[2]

        t_hat = y[:2] / yr
        contact_mode = self.compute_contact_mode(sap_info[i_p].mu, sap_info[i_p].mu_hat, yr, yn)
        sap_info[i_p].gamma.fill(0.0)
        sap_info[i_p].G.fill(0.0)
        if contact_mode == ContactMode.STICK:
            sap_info[i_p].gamma = y
            sap_info[i_p].G[0, 0] = sap_info[i_p].Rt_inv
            sap_info[i_p].G[1, 1] = sap_info[i_p].Rt_inv
            sap_info[i_p].G[2, 2] = sap_info[i_p].Rn_inv
        elif contact_mode == ContactMode.SLIDE:
            gn = (yn + sap_info[i_p].mu_hat * yr) * sap_info[i_p].mu_factor
            gt = sap_info[i_p].mu * gn * t_hat
            sap_info[i_p].gamma = qd.Vector([gt[0], gt[1], gn])
            P = t_hat.outer_product(t_hat)
            Pperp = qd.Matrix.identity(gs.qd_float, 2) - P
            dgt_dyt = sap_info[i_p].mu * (gn / yr * Pperp + sap_info[i_p].mu_hat * sap_info[i_p].mu_factor * P)
            dgt_dyn = sap_info[i_p].mu * sap_info[i_p].mu_factor * t_hat
            dgn_dyt = sap_info[i_p].mu_hat * sap_info[i_p].mu_factor * t_hat
            dgn_dyn = sap_info[i_p].mu_factor

            sap_info[i_p].G[:2, :2] = dgt_dyt * sap_info[i_p].Rt_inv
            sap_info[i_p].G[:2, 2] = dgt_dyn * sap_info[i_p].Rn_inv
            sap_info[i_p].G[2, :2] = dgn_dyt * sap_info[i_p].Rt_inv
            sap_info[i_p].G[2, 2] = dgn_dyn * sap_info[i_p].Rn_inv
        else:  # No contact
            pass

    @qd.func
    def compute_contact_energy_gamma_G(self, sap_info, i_p, vc):
        self.compute_contact_gamma_G(sap_info, i_p, vc)
        R_gamma = sap_info[i_p].gamma
        R_gamma[0] *= sap_info[i_p].Rt
        R_gamma[1] *= sap_info[i_p].Rt
        R_gamma[2] *= sap_info[i_p].Rn
        sap_info[i_p].energy = 0.5 * sap_info[i_p].gamma.dot(R_gamma)

    @qd.func
    def compute_contact_energy(self, sap_info, i_p, vc):
        y = qd.Vector([0.0, 0.0, sap_info[i_p].vn_hat]) - vc
        y[0] *= sap_info[i_p].Rt_inv
        y[1] *= sap_info[i_p].Rt_inv
        y[2] *= sap_info[i_p].Rn_inv
        yr = y[:2].norm(gs.EPS)
        yn = y[2]

        t_hat = y[:2] / yr
        contact_mode = self.compute_contact_mode(sap_info[i_p].mu, sap_info[i_p].mu_hat, yr, yn)
        sap_info[i_p].gamma.fill(0.0)
        if contact_mode == ContactMode.STICK:
            sap_info[i_p].gamma = y
        elif contact_mode == ContactMode.SLIDE:
            gn = (yn + sap_info[i_p].mu_hat * yr) * sap_info[i_p].mu_factor
            gt = sap_info[i_p].mu * gn * t_hat
            sap_info[i_p].gamma = qd.Vector([gt[0], gt[1], gn])
        else:  # No contact
            pass

        R_gamma = sap_info[i_p].gamma
        R_gamma[0] *= sap_info[i_p].Rt
        R_gamma[1] *= sap_info[i_p].Rt
        R_gamma[2] *= sap_info[i_p].Rn
        sap_info[i_p].energy = 0.5 * sap_info[i_p].gamma.dot(R_gamma)

    @qd.func
    def compute_contact_mode(self, mu, mu_hat, yr, yn):
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
    def compute_contact_regularization(self, sap_info, i_p, w_rms, time_step):
        beta_factor = self.coupler._sap_beta**2 / (4.0 * qd.math.pi**2)
        k = sap_info[i_p].k
        Rn = max(beta_factor * w_rms, 1.0 / (time_step * k * (time_step + self.coupler._sap_taud)))
        Rt = self.coupler._sap_sigma * w_rms
        vn_hat = -sap_info[i_p].phi0 / (time_step + self.coupler._sap_taud)
        sap_info[i_p].Rn = Rn
        sap_info[i_p].Rt = Rt
        sap_info[i_p].Rn_inv = 1.0 / Rn
        sap_info[i_p].Rt_inv = 1.0 / Rt
        sap_info[i_p].vn_hat = vn_hat
        sap_info[i_p].mu_hat = sap_info[i_p].mu * Rt * sap_info[i_p].Rn_inv
        sap_info[i_p].mu_factor = 1.0 / (1.0 + sap_info[i_p].mu * sap_info[i_p].mu_hat)


@qd.data_oriented
class RigidContactHandler(BaseContactHandler):
    def __init__(self, simulator: "Simulator") -> None:
        super().__init__(simulator)
        self.rigid_solver = self.sim.rigid_solver

    # FIXME This function is similar to the one in constraint_solver.py:add_collision_constraints.
    # Consider refactoring, using better naming, and removing while.
    @qd.func
    def compute_jacobian(
        self, links_info: array_class.LinksInfo, dofs_state: array_class.DofsState, links_state: array_class.LinksState
    ):
        self.Jt.fill(0.0)
        for i_p in range(self.n_contact_pairs[None]):
            link = self.contact_pairs[i_p].link_idx
            i_b = self.contact_pairs[i_p].batch_idx
            while link > -1:
                link_maybe_batch = [link, i_b] if qd.static(self.rigid_solver._options.batch_links_info) else link
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
    def compute_gradient_hessian_diag(self):
        sap_info = qd.static(self.contact_pairs.sap_info)
        for i_p in range(self.n_contact_pairs[None]):
            vc = self.compute_contact_velocity(i_p)
            self.compute_contact_gamma_G(sap_info, i_p, vc)
            self.add_Jt_x(self.coupler.rigid_state_dof.gradient, i_p, -sap_info[i_p].gamma)
            self.add_Jt_x(self.coupler.rigid_state_dof.impulse, i_p, sap_info[i_p].gamma)

    @qd.func
    def compute_pcg_matrix_vector_product(self):
        sap_info = qd.static(self.contact_pairs.sap_info)
        for i_p in range(self.n_contact_pairs[None]):
            # Jt @ G @ J @ p
            Jp = self.compute_Jx(i_p, self.coupler.pcg_rigid_state_dof.p)
            GJp = sap_info[i_p].G @ Jp
            self.add_Jt_x(self.coupler.pcg_rigid_state_dof.Ap, i_p, GJp)

    @qd.func
    def compute_contact_velocity(self, i_p):
        """
        Compute the contact velocity in the contact frame.
        """
        return self.compute_Jx(i_p, self.coupler.rigid_state_dof.v)

    @qd.func
    def prepare_search_direction_data(self):
        sap_info = qd.static(self.contact_pairs.sap_info)
        for i_p in qd.ndrange(self.n_contact_pairs[None]):
            i_b = self.contact_pairs[i_p].batch_idx
            if self.coupler.batch_linesearch_active[i_b]:
                sap_info[i_p].dvc = self.compute_Jx(i_p, self.coupler.pcg_rigid_state_dof.x)

    @qd.func
    def compute_delassus_world_frame(self, entities_info: array_class.EntitiesInfo, rigid_info: array_class.RigidInfo):
        self.coupler.rigid_solve_jacobian(
            self.Jt,
            self.M_inv_Jt,
            self.n_contact_pairs[None],
            self.contact_pairs.batch_idx,
            3,
            entities_info=entities_info,
            rigid_info=rigid_info,
        )
        self.W.fill(0.0)
        for i_p, i_d, i, j in qd.ndrange(self.n_contact_pairs[None], self.rigid_solver.n_dofs, 3, 3):
            self.W[i_p][i, j] += self.M_inv_Jt[i_p, i_d][i] * self.Jt[i_p, i_d][j]

    @qd.func
    def compute_delassus(self, i_p):
        return self.W[i_p]

    @qd.func
    def compute_Jx(self, i_p, x):
        pairs = qd.static(self.contact_pairs)
        i_b = pairs[i_p].batch_idx
        Jx = qd.Vector.zero(gs.qd_float, 3)
        for i in range(self.rigid_solver.n_dofs):
            Jx = Jx + self.Jt[i_p, i] * x[i_b, i]
        return Jx

    @qd.func
    def add_Jt_x(self, y, i_p, x):
        pairs = qd.static(self.contact_pairs)
        i_b = pairs[i_p].batch_idx
        for i in range(self.rigid_solver.n_dofs):
            y[i_b, i] += self.Jt[i_p, i].dot(x)


@qd.data_oriented
class RigidRigidContactHandler(RigidContactHandler):
    def __init__(self, simulator: "Simulator") -> None:
        super().__init__(simulator)

    @qd.func
    def compute_jacobian(
        self, links_info: array_class.LinksInfo, dofs_state: array_class.DofsState, links_state: array_class.LinksState
    ):
        self.Jt.fill(0.0)
        pairs = qd.static(self.contact_pairs)
        for i_p in range(self.n_contact_pairs[None]):
            i_b = pairs[i_p].batch_idx
            link = pairs[i_p].link_idx0
            while link > -1:
                link_maybe_batch = [link, i_b] if qd.static(self.rigid_solver._options.batch_links_info) else link
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
                link_maybe_batch = [link, i_b] if qd.static(self.rigid_solver._options.batch_links_info) else link
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
    def compute_delassus(self, i_p):
        pairs = qd.static(self.contact_pairs)
        world = qd.Matrix.cols([pairs[i_p].tangent0, pairs[i_p].tangent1, pairs[i_p].normal])
        return world.transpose() @ self.W[i_p] @ world

    @qd.func
    def compute_Jx(self, i_p, x):
        pairs = qd.static(self.contact_pairs)
        i_b = pairs[i_p].batch_idx
        Jx = qd.Vector.zero(gs.qd_float, 3)
        for i in range(self.rigid_solver.n_dofs):
            Jx = Jx + self.Jt[i_p, i] * x[i_b, i]
        Jx = qd.Vector([Jx.dot(pairs[i_p].tangent0), Jx.dot(pairs[i_p].tangent1), Jx.dot(pairs[i_p].normal)])
        return Jx

    @qd.func
    def add_Jt_x(self, y, i_p, x):
        pairs = qd.static(self.contact_pairs)
        i_b = pairs[i_p].batch_idx
        world = qd.Matrix.cols([pairs[i_p].tangent0, pairs[i_p].tangent1, pairs[i_p].normal])
        x_ = world @ x
        for i in range(self.rigid_solver.n_dofs):
            y[i_b, i] += self.Jt[i_p, i].dot(x_)


@qd.data_oriented
class FEMContactHandler(BaseContactHandler):
    def __init__(self, simulator: "Simulator") -> None:
        super().__init__(simulator)
        self.fem_solver = simulator.fem_solver

    @qd.func
    def compute_gradient_hessian_diag(self):
        sap_info = qd.static(self.contact_pairs.sap_info)
        for i_p in range(self.n_contact_pairs[None]):
            vc = self.compute_Jx(i_p, self.coupler.fem_state_v.v)
            self.compute_contact_gamma_G(sap_info, i_p, vc)
            self.add_Jt_x(self.coupler.fem_state_v.gradient, i_p, -sap_info[i_p].gamma)
            self.add_Jt_x(self.coupler.fem_state_v.impulse, i_p, sap_info[i_p].gamma)
            self.add_Jt_A_J_diag3x3(self.coupler.pcg_fem_state_v.diag3x3, i_p, sap_info[i_p].G)

    @qd.func
    def prepare_search_direction_data(self):
        sap_info = qd.static(self.contact_pairs.sap_info)
        for i_p in qd.ndrange(self.n_contact_pairs[None]):
            i_b = self.contact_pairs[i_p].batch_idx
            if self.coupler.batch_linesearch_active[i_b]:
                sap_info[i_p].dvc = self.compute_Jx(i_p, self.coupler.pcg_fem_state_v.x)

    @qd.func
    def compute_pcg_matrix_vector_product(self):
        sap_info = qd.static(self.contact_pairs.sap_info)
        for i_p in range(self.n_contact_pairs[None]):
            # Jt @ G @ J @ p
            x = self.compute_Jx(i_p, self.coupler.pcg_fem_state_v.p)
            x = sap_info[i_p].G @ x
            self.add_Jt_x(self.coupler.pcg_fem_state_v.Ap, i_p, x)

    @qd.func
    def compute_contact_velocity(self, i_p):
        """
        Compute the contact velocity in the contact frame.
        """
        return self.compute_Jx(i_p, self.coupler.fem_state_v.v)


@qd.data_oriented
class RigidFEMContactHandler(RigidContactHandler):
    def __init__(self, simulator: "Simulator") -> None:
        super().__init__(simulator)
        self.fem_solver = simulator.fem_solver

    @qd.func
    def compute_gradient_hessian_diag(self):
        sap_info = qd.static(self.contact_pairs.sap_info)
        for i_p in range(self.n_contact_pairs[None]):
            vc = self.compute_Jx(i_p, self.coupler.fem_state_v.v, self.coupler.rigid_state_dof.v)
            self.compute_contact_gamma_G(sap_info, i_p, vc)
            self.add_Jt_x(
                self.coupler.fem_state_v.gradient, self.coupler.rigid_state_dof.gradient, i_p, -sap_info[i_p].gamma
            )
            self.add_Jt_x(
                self.coupler.fem_state_v.impulse, self.coupler.rigid_state_dof.impulse, i_p, sap_info[i_p].gamma
            )
            self.add_Jt_A_J_diag3x3(self.coupler.pcg_fem_state_v.diag3x3, i_p, sap_info[i_p].G)

    @qd.func
    def prepare_search_direction_data(self):
        sap_info = qd.static(self.contact_pairs.sap_info)
        for i_p in qd.ndrange(self.n_contact_pairs[None]):
            i_b = self.contact_pairs[i_p].batch_idx
            if self.coupler.batch_linesearch_active[i_b]:
                sap_info[i_p].dvc = self.compute_Jx(
                    i_p, self.coupler.pcg_fem_state_v.x, self.coupler.pcg_rigid_state_dof.x
                )

    @qd.func
    def compute_pcg_matrix_vector_product(self):
        sap_info = qd.static(self.contact_pairs.sap_info)
        for i_p in range(self.n_contact_pairs[None]):
            # Jt @ G @ J @ p
            x = self.compute_Jx(i_p, self.coupler.pcg_fem_state_v.p, self.coupler.pcg_rigid_state_dof.p)
            x = sap_info[i_p].G @ x
            self.add_Jt_x(self.coupler.pcg_fem_state_v.Ap, self.coupler.pcg_rigid_state_dof.Ap, i_p, x)

    @qd.func
    def compute_contact_velocity(self, i_p):
        """
        Compute the contact velocity in the contact frame.
        """
        return self.compute_Jx(i_p, self.coupler.fem_state_v.v, self.coupler.rigid_state_dof.v)


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

    def __init__(self, simulator: "Simulator", eps: float = 1e-10) -> None:
        super().__init__(simulator)
        self.name = "FEMFloorTetContactHandler"
        self.fem_solver = self.sim.fem_solver
        self.eps = eps
        self.eps = eps
        self.contact_candidate_type = qd.types.struct(
            batch_idx=gs.qd_int,  # batch index
            geom_idx=gs.qd_int,  # index of the FEM element
            intersection_code=gs.qd_int,  # intersection code for the element
            distance=gs.qd_vec4,  # distance vector for the element
        )
        self.n_contact_candidates = qd.field(gs.qd_int, shape=())
        self.max_contact_candidates = self.fem_solver.n_surface_elements * self.fem_solver._B
        self.contact_candidates = self.contact_candidate_type.field(shape=(self.max_contact_candidates,))

        self.contact_pair_type = qd.types.struct(
            batch_idx=gs.qd_int,  # batch index
            geom_idx=gs.qd_int,  # index of the FEM element
            barycentric=gs.qd_vec4,  # barycentric coordinates of the contact point
            contact_pos=gs.qd_vec3,  # contact position
            sap_info=self.sap_contact_info_type,  # contact info
        )
        self.max_contact_pairs = self.fem_solver.n_surface_elements * self.fem_solver._B
        self.contact_pairs = self.contact_pair_type.field(shape=(self.max_contact_pairs,))

    @qd.func
    def detection(
        self,
        f: qd.i32,
        links_info: array_class.LinksInfo,
        verts_info: array_class.VertsInfo,
        faces_info: array_class.FacesInfo,
        free_verts_state: array_class.VertsState,
        fixed_verts_state: array_class.VertsState,
        geoms_info: array_class.GeomsInfo,
    ):
        overflow = False
        # Compute contact pairs
        self.n_contact_candidates[None] = 0
        # TODO Check surface element only instead of all elements
        for i_b, i_e in qd.ndrange(self.fem_solver._B, self.fem_solver.n_elements):
            intersection_code = qd.int32(0)
            distance = qd.Vector.zero(gs.qd_float, 4)
            for i in qd.static(range(4)):
                i_v = self.fem_solver.elements_i[i_e].el2v[i]
                pos_v = self.fem_solver.elements_v[f, i_v, i_b].pos
                distance[i] = pos_v.z - self.fem_solver.floor_height
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
                    overflow = True

        sap_info = qd.static(self.contact_pairs.sap_info)
        self.n_contact_pairs[None] = 0
        # Compute pair from candidates
        result_count = qd.min(self.n_contact_candidates[None], self.max_contact_candidates)
        for i_c in range(result_count):
            candidate = self.contact_candidates[i_c]
            i_b = candidate.batch_idx
            i_e = candidate.geom_idx
            intersection_code = candidate.intersection_code
            intersected_edges = self.coupler.MarchingTetsEdgeTable[intersection_code]

            tet_vertices = qd.Matrix.zero(gs.qd_float, 3, 4)  # 4 vertices
            tet_pressures = qd.Vector.zero(gs.qd_float, 4)  # pressures at the vertices
            for i in qd.static(range(4)):
                i_v = self.fem_solver.elements_i[i_e].el2v[i]
                tet_vertices[:, i] = self.fem_solver.elements_v[f, i_v, i_b].pos
                tet_pressures[i] = self.coupler.fem_pressure[i_v]

            polygon_vertices = qd.Matrix.zero(gs.qd_float, 3, 4)  # 3 or 4 vertices
            total_area = gs.EPS  # avoid division by zero
            total_area_weighted_centroid = qd.Vector.zero(gs.qd_float, 3)
            for i in qd.static(range(4)):
                if intersected_edges[i] >= 0:
                    edge = self.coupler.TetEdges[intersected_edges[i]]
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

            deformable_g = self.coupler._hydroelastic_stiffness
            rigid_g = self.coupler.fem_pressure_gradient[i_b, i_e].z
            # TODO A better way to handle corner cases where pressure and pressure gradient are ill defined
            if total_area < self.eps or rigid_g < self.eps:
                continue
            g = 1.0 / (1.0 / deformable_g + 1.0 / rigid_g)  # harmonic average
            rigid_k = total_area * g
            rigid_phi0 = -pressure / g
            if rigid_k < self.eps or rigid_phi0 > self.eps:
                continue
            i_p = qd.atomic_add(self.n_contact_pairs[None], 1)
            if i_p < self.max_contact_pairs:
                self.contact_pairs[i_p].batch_idx = i_b
                self.contact_pairs[i_p].geom_idx = i_e
                self.contact_pairs[i_p].barycentric = barycentric
                sap_info[i_p].k = rigid_k
                sap_info[i_p].phi0 = rigid_phi0
                sap_info[i_p].mu = self.fem_solver.elements_i[i_e].friction_mu
            else:
                overflow = True

        return overflow

    @qd.func
    def compute_Jx(self, i_p, x):
        """
        Compute the contact Jacobian J times a vector x.
        """
        i_b = self.contact_pairs[i_p].batch_idx
        i_g = self.contact_pairs[i_p].geom_idx
        Jx = qd.Vector.zero(gs.qd_float, 3)
        for i in qd.static(range(4)):
            i_v = self.fem_solver.elements_i[i_g].el2v[i]
            Jx += self.contact_pairs[i_p].barycentric[i] * x[i_b, i_v]
        return Jx

    @qd.func
    def add_Jt_x(self, y, i_p, x):
        i_b = self.contact_pairs[i_p].batch_idx
        i_g = self.contact_pairs[i_p].geom_idx
        for i in qd.static(range(4)):
            i_v = self.fem_solver.elements_i[i_g].el2v[i]
            if qd.static(self.fem_solver._enable_vertex_constraints):
                if not self.fem_solver.vertex_constraints.is_constrained[i_v, i_b]:
                    y[i_b, i_v] += self.contact_pairs[i_p].barycentric[i] * x
            else:
                y[i_b, i_v] += self.contact_pairs[i_p].barycentric[i] * x

    @qd.func
    def add_Jt_A_J_diag3x3(self, y, i_p, A):
        i_b = self.contact_pairs[i_p].batch_idx
        i_g = self.contact_pairs[i_p].geom_idx
        for i in qd.static(range(4)):
            i_v = self.fem_solver.elements_i[i_g].el2v[i]
            if qd.static(self.fem_solver._enable_vertex_constraints):
                if not self.fem_solver.vertex_constraints.is_constrained[i_v, i_b]:
                    y[i_b, i_v] += self.contact_pairs[i_p].barycentric[i] ** 2 * A
            else:
                y[i_b, i_v] += self.contact_pairs[i_p].barycentric[i] ** 2 * A

    @qd.func
    def compute_delassus(self, i_p):
        dt2_inv = 1.0 / self.sim._substep_dt**2
        i_b = self.contact_pairs[i_p].batch_idx
        i_g = self.contact_pairs[i_p].geom_idx
        W = qd.Matrix.zero(gs.qd_float, 3, 3)
        # W = sum (JA^-1J^T)
        # With floor, J is Identity times the barycentric coordinates
        for i in qd.static(range(4)):
            i_v = self.fem_solver.elements_i[i_g].el2v[i]
            W += self.contact_pairs[i_p].barycentric[i] ** 2 * dt2_inv * self.fem_solver.pcg_state_v[i_b, i_v].prec
        return W


@qd.data_oriented
class FEMSelfTetContactHandler(FEMContactHandler):
    """
    Class for handling self-contact between tetrahedral elements in a simulation using hydroelastic model.

    This class extends the FEMContact class and provides methods for detecting self-contact
    between tetrahedral elements, computing contact pairs, and managing contact-related computations.
    """

    def __init__(self, simulator: "Simulator", eps: float = 1e-10) -> None:
        super().__init__(simulator)
        self.name = "FEMSelfTetContactHandler"
        self.eps = eps
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
        self.max_contact_candidates = self.fem_solver.n_surface_elements * self.fem_solver._B * 8
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
            sap_info=self.sap_contact_info_type,  # contact info
        )
        self.max_contact_pairs = self.fem_solver.n_surface_elements * self.fem_solver._B
        self.contact_pairs = self.contact_pair_type.field(shape=(self.max_contact_pairs,))

    @qd.func
    def compute_candidates(self, f: qd.i32):
        overflow = False
        self.n_contact_candidates[None] = 0
        result_count = qd.min(
            self.coupler.fem_surface_tet_bvh.query_result_count[None],
            self.coupler.fem_surface_tet_bvh.max_query_results,
        )
        for i_r in range(result_count):
            i_b, i_sa, i_sq = self.coupler.fem_surface_tet_bvh.query_result[i_r]
            i_a = self.fem_solver.surface_elements[i_sa]
            i_q = self.fem_solver.surface_elements[i_sq]
            i_v0 = self.fem_solver.elements_i[i_a].el2v[0]
            i_v1 = self.fem_solver.elements_i[i_q].el2v[0]
            x0 = self.fem_solver.elements_v[f, i_v0, i_b].pos
            x1 = self.fem_solver.elements_v[f, i_v1, i_b].pos
            p0 = self.coupler.fem_pressure[i_v0]
            p1 = self.coupler.fem_pressure[i_v1]
            g0 = self.coupler.fem_pressure_gradient[i_b, i_a]
            g1 = self.coupler.fem_pressure_gradient[i_b, i_q]
            g0_norm = g0.norm()
            g1_norm = g1.norm()
            if g0_norm < gs.EPS or g1_norm < gs.EPS:
                continue
            # Calculate the isosurface, i.e. equal pressure plane defined by x and normal
            # Solve for p0 + g0.dot(x - x0) = p1 + g1.dot(x - x1)
            normal = g0 - g1
            magnitude = normal.norm()
            if magnitude < gs.EPS:
                continue
            normal /= magnitude
            b = p1 - p0 - g1.dot(x1) + g0.dot(x0)
            x = b / magnitude * normal
            # Check that the normal is pointing along g0 and against g1, some allowance as used in Drake
            threshold = qd.static(np.cos(np.pi * 5.0 / 8.0))
            if normal.dot(g0) < threshold * g0_norm or normal.dot(g1) > -threshold * g1_norm:
                continue
            intersection_code0 = qd.int32(0)
            distance0 = qd.Vector.zero(gs.qd_float, 4)
            intersection_code1 = qd.int32(0)
            distance1 = qd.Vector.zero(gs.qd_float, 4)
            for i in qd.static(range(4)):
                i_v = self.fem_solver.elements_i[i_a].el2v[i]
                pos_v = self.fem_solver.elements_v[f, i_v, i_b].pos
                distance0[i] = (pos_v - x).dot(normal)  # signed distance
                if distance0[i] > 0.0:
                    intersection_code0 |= 1 << i
            for i in qd.static(range(4)):
                i_v = self.fem_solver.elements_i[i_q].el2v[i]
                pos_v = self.fem_solver.elements_v[f, i_v, i_b].pos
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
                overflow = True
        return overflow

    @qd.func
    def compute_pairs(self, i_step: qd.i32):
        """
        Computes the FEM self contact pairs and their properties.

        Intersection code reference:
        https://github.com/RobotLocomotion/drake/blob/8c3a249184ed09f0faab3c678536d66d732809ce/geometry/proximity/field_intersection.cc#L87
        """
        overflow = False
        sap_info = qd.static(self.contact_pairs.sap_info)
        normal_signs = qd.Vector([1.0, -1.0, 1.0, -1.0], dt=gs.qd_float)  # make normal point outward
        self.n_contact_pairs[None] = 0
        result_count = qd.min(self.n_contact_candidates[None], self.max_contact_candidates)
        for i_c in range(result_count):
            i_b = self.contact_candidates[i_c].batch_idx
            i_e0 = self.contact_candidates[i_c].geom_idx0
            i_e1 = self.contact_candidates[i_c].geom_idx1
            intersection_code0 = self.contact_candidates[i_c].intersection_code0
            distance0 = self.contact_candidates[i_c].distance0
            intersected_edges0 = self.coupler.MarchingTetsEdgeTable[intersection_code0]

            tet_vertices0 = qd.Matrix.zero(gs.qd_float, 3, 4)  # 4 vertices of tet 0
            tet_pressures0 = qd.Vector.zero(gs.qd_float, 4)  # pressures at the vertices of tet 0
            tet_vertices1 = qd.Matrix.zero(gs.qd_float, 3, 4)  # 4 vertices of tet 1
            for i in qd.static(range(4)):
                i_v = self.fem_solver.elements_i[i_e0].el2v[i]
                tet_vertices0[:, i] = self.fem_solver.elements_v[i_step, i_v, i_b].pos
                tet_pressures0[i] = self.coupler.fem_pressure[i_v]
            for i in qd.static(range(4)):
                i_v = self.fem_solver.elements_i[i_e1].el2v[i]
                tet_vertices1[:, i] = self.fem_solver.elements_v[i_step, i_v, i_b].pos

            polygon_vertices = qd.Matrix.zero(gs.qd_float, 3, 8)  # maximum 8 vertices
            polygon_n_vertices = gs.qd_int(0)
            clipped_vertices = qd.Matrix.zero(gs.qd_float, 3, 8)  # maximum 8 vertices
            clipped_n_vertices = gs.qd_int(0)
            for i in range(4):
                if intersected_edges0[i] >= 0:
                    edge = self.coupler.TetEdges[intersected_edges0[i]]
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

            if total_area < self.eps:
                continue
            centroid = total_area_weighted_centroid / total_area
            barycentric0 = tet_barycentric(centroid, tet_vertices0)
            barycentric1 = tet_barycentric(centroid, tet_vertices1)
            tangent0 = polygon_vertices[:, 0] - centroid
            tangent0 /= tangent0.norm()
            tangent1 = self.contact_candidates[i_c].normal.cross(tangent0)

            pressure = barycentric0.dot(tet_pressures0)
            g0 = self.coupler.fem_pressure_gradient[i_b, i_e0].dot(self.contact_candidates[i_c].normal)
            g1 = -self.coupler.fem_pressure_gradient[i_b, i_e1].dot(self.contact_candidates[i_c].normal)
            # FIXME This is an approximated value, different from Drake, which actually calculates the distance
            deformable_phi0 = -pressure / g0 - pressure / g1

            if deformable_phi0 > gs.EPS:
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

                deformable_g = self.coupler._hydroelastic_stiffness
                deformable_k = total_area * deformable_g
                sap_info[i_p].k = deformable_k
                sap_info[i_p].phi0 = deformable_phi0
                sap_info[i_p].mu = qd.sqrt(
                    self.fem_solver.elements_i[i_e0].friction_mu * self.fem_solver.elements_i[i_e1].friction_mu
                )
            else:
                overflow = True
        return overflow

    @qd.func
    def detection(
        self,
        f: qd.i32,
        links_info: array_class.LinksInfo,
        verts_info: array_class.VertsInfo,
        faces_info: array_class.FacesInfo,
        free_verts_state: array_class.VertsState,
        fixed_verts_state: array_class.VertsState,
        geoms_info: array_class.GeomsInfo,
    ):
        overflow = False
        overflow |= self.coupler.fem_surface_tet_bvh.query(self.coupler.fem_surface_tet_aabb.aabbs)
        overflow |= self.compute_candidates(f)
        overflow |= self.compute_pairs(f)
        return overflow

    @qd.func
    def compute_Jx(self, i_p, x):
        """
        Compute the contact Jacobian J times a vector x.
        """
        i_b = self.contact_pairs[i_p].batch_idx
        i_g0 = self.contact_pairs[i_p].geom_idx0
        i_g1 = self.contact_pairs[i_p].geom_idx1
        Jx = qd.Vector.zero(gs.qd_float, 3)
        for i in qd.static(range(4)):
            i_v = self.fem_solver.elements_i[i_g0].el2v[i]
            Jx += self.contact_pairs[i_p].barycentric0[i] * x[i_b, i_v]
        for i in qd.static(range(4)):
            i_v = self.fem_solver.elements_i[i_g1].el2v[i]
            Jx -= self.contact_pairs[i_p].barycentric1[i] * x[i_b, i_v]
        return qd.Vector(
            [
                Jx.dot(self.contact_pairs[i_p].tangent0),
                Jx.dot(self.contact_pairs[i_p].tangent1),
                Jx.dot(self.contact_pairs[i_p].normal),
            ]
        )

    @qd.func
    def add_Jt_x(self, y, i_p, x):
        i_b = self.contact_pairs[i_p].batch_idx
        i_g0 = self.contact_pairs[i_p].geom_idx0
        i_g1 = self.contact_pairs[i_p].geom_idx1
        world = qd.Matrix.cols(
            [self.contact_pairs[i_p].tangent0, self.contact_pairs[i_p].tangent1, self.contact_pairs[i_p].normal]
        )
        x_ = world @ x
        for i in qd.static(range(4)):
            i_v = self.fem_solver.elements_i[i_g0].el2v[i]
            if qd.static(self.fem_solver._enable_vertex_constraints):
                if not self.fem_solver.vertex_constraints.is_constrained[i_v, i_b]:
                    y[i_b, i_v] += self.contact_pairs[i_p].barycentric0[i] * x_
            else:
                y[i_b, i_v] += self.contact_pairs[i_p].barycentric0[i] * x_
        for i in qd.static(range(4)):
            i_v = self.fem_solver.elements_i[i_g1].el2v[i]
            if qd.static(self.fem_solver._enable_vertex_constraints):
                if not self.fem_solver.vertex_constraints.is_constrained[i_v, i_b]:
                    y[i_b, i_v] -= self.contact_pairs[i_p].barycentric1[i] * x_
            else:
                y[i_b, i_v] -= self.contact_pairs[i_p].barycentric1[i] * x_

    @qd.func
    def add_Jt_A_J_diag3x3(self, y, i_p, A):
        i_b = self.contact_pairs[i_p].batch_idx
        i_g0 = self.contact_pairs[i_p].geom_idx0
        i_g1 = self.contact_pairs[i_p].geom_idx1
        world = qd.Matrix.cols(
            [self.contact_pairs[i_p].tangent0, self.contact_pairs[i_p].tangent1, self.contact_pairs[i_p].normal]
        )
        B_ = world @ A @ world.transpose()
        for i in qd.static(range(4)):
            i_v = self.fem_solver.elements_i[i_g0].el2v[i]
            if qd.static(self.fem_solver._enable_vertex_constraints):
                if not self.fem_solver.vertex_constraints.is_constrained[i_v, i_b]:
                    y[i_b, i_v] += self.contact_pairs[i_p].barycentric0[i] ** 2 * B_
            else:
                y[i_b, i_v] += self.contact_pairs[i_p].barycentric0[i] ** 2 * B_
        for i in qd.static(range(4)):
            i_v = self.fem_solver.elements_i[i_g1].el2v[i]
            if qd.static(self.fem_solver._enable_vertex_constraints):
                if not self.fem_solver.vertex_constraints.is_constrained[i_v, i_b]:
                    y[i_b, i_v] += self.contact_pairs[i_p].barycentric1[i] ** 2 * B_
            else:
                y[i_b, i_v] += self.contact_pairs[i_p].barycentric1[i] ** 2 * B_

    @qd.func
    def compute_delassus(self, i_p):
        dt2_inv = 1.0 / self.sim._substep_dt**2
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
            i_v = self.fem_solver.elements_i[i_g0].el2v[i]
            W += self.contact_pairs[i_p].barycentric0[i] ** 2 * dt2_inv * self.fem_solver.pcg_state_v[i_b, i_v].prec
        for i in qd.static(range(4)):
            i_v = self.fem_solver.elements_i[i_g1].el2v[i]
            W += self.contact_pairs[i_p].barycentric1[i] ** 2 * dt2_inv * self.fem_solver.pcg_state_v[i_b, i_v].prec
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
        super().__init__(simulator)
        self.name = "FEMFloorVertContactHandler"
        self.fem_solver = self.sim.fem_solver

        self.contact_pair_type = qd.types.struct(
            batch_idx=gs.qd_int,  # batch index
            geom_idx=gs.qd_int,  # index of the vertex
            contact_pos=gs.qd_vec3,  # contact position
            sap_info=self.sap_contact_info_type,  # contact info
        )
        self.max_contact_pairs = self.fem_solver.n_surface_elements * self.fem_solver._B
        self.contact_pairs = self.contact_pair_type.field(shape=(self.max_contact_pairs,))

    @qd.func
    def detection(
        self,
        f: qd.i32,
        links_info: array_class.LinksInfo,
        verts_info: array_class.VertsInfo,
        faces_info: array_class.FacesInfo,
        free_verts_state: array_class.VertsState,
        fixed_verts_state: array_class.VertsState,
        geoms_info: array_class.GeomsInfo,
    ):
        overflow = False
        sap_info = qd.static(self.contact_pairs.sap_info)
        # Compute contact pairs
        self.n_contact_pairs[None] = 0
        for i_b, i_sv in qd.ndrange(self.fem_solver._B, self.fem_solver.n_surface_vertices):
            i_v = self.fem_solver.surface_vertices[i_sv]
            pos_v = self.fem_solver.elements_v[f, i_v, i_b].pos
            distance = pos_v.z - self.fem_solver.floor_height
            if distance > 0.0:
                continue
            i_p = qd.atomic_add(self.n_contact_pairs[None], 1)
            if i_p < self.max_contact_pairs:
                self.contact_pairs[i_p].batch_idx = i_b
                self.contact_pairs[i_p].geom_idx = i_v
                sap_info[i_p].k = self.coupler._point_contact_stiffness * self.fem_solver.surface_vert_mass[i_v]
                sap_info[i_p].phi0 = distance
                sap_info[i_p].mu = self.fem_solver.elements_v_info[i_v].friction_mu
            else:
                overflow = True
        return overflow

    @qd.func
    def compute_Jx(self, i_p, x):
        """
        Compute the contact Jacobian J times a vector x.
        """
        i_b = self.contact_pairs[i_p].batch_idx
        i_g = self.contact_pairs[i_p].geom_idx
        Jx = x[i_b, i_g]
        return Jx

    @qd.func
    def add_Jt_x(self, y, i_p, x):
        i_b = self.contact_pairs[i_p].batch_idx
        i_g = self.contact_pairs[i_p].geom_idx
        if qd.static(self.fem_solver._enable_vertex_constraints):
            if not self.fem_solver.vertex_constraints.is_constrained[i_g, i_b]:
                y[i_b, i_g] += x
        else:
            y[i_b, i_g] += x

    @qd.func
    def add_Jt_A_J_diag3x3(self, y, i_p, A):
        i_b = self.contact_pairs[i_p].batch_idx
        i_g = self.contact_pairs[i_p].geom_idx
        if qd.static(self.fem_solver._enable_vertex_constraints):
            if not self.fem_solver.vertex_constraints.is_constrained[i_g, i_b]:
                y[i_b, i_g] += A
        else:
            y[i_b, i_g] += A

    @qd.func
    def compute_delassus(self, i_p):
        dt2_inv = 1.0 / self.sim._substep_dt**2
        i_b = self.contact_pairs[i_p].batch_idx
        i_g = self.contact_pairs[i_p].geom_idx
        # W = sum (JA^-1J^T)
        # With floor, J is Identity
        W = self.fem_solver.pcg_state_v[i_b, i_g].prec * dt2_inv
        return W


@qd.data_oriented
class RigidFloorVertContactHandler(RigidContactHandler):
    def __init__(self, simulator: "Simulator") -> None:
        super().__init__(simulator)
        self.name = "RigidFloorVertContactHandler"
        self.rigid_solver = self.sim.rigid_solver
        self.floor_height = self.sim.fem_solver.floor_height
        self.contact_pair_type = qd.types.struct(
            batch_idx=gs.qd_int,  # batch index
            link_idx=gs.qd_int,  # index of the link
            contact_pos=gs.qd_vec3,  # contact position
            sap_info=self.sap_contact_info_type,  # contact info
        )
        self.max_contact_pairs = self.rigid_solver.n_free_verts * self.sim._B
        self.contact_pairs = self.contact_pair_type.field(shape=(self.max_contact_pairs,))
        self.Jt = qd.field(gs.qd_vec3, shape=(self.max_contact_pairs, self.rigid_solver.n_dofs))
        self.M_inv_Jt = qd.field(gs.qd_vec3, shape=(self.max_contact_pairs, self.rigid_solver.n_dofs))
        self.W = qd.field(gs.qd_mat3, shape=(self.max_contact_pairs,))

    @qd.func
    def detection(
        self,
        f: qd.i32,
        links_info: array_class.LinksInfo,
        verts_info: array_class.VertsInfo,
        faces_info: array_class.FacesInfo,
        free_verts_state: array_class.VertsState,
        fixed_verts_state: array_class.VertsState,
        geoms_info: array_class.GeomsInfo,
    ):
        overflow = False
        sap_info = qd.static(self.contact_pairs.sap_info)
        C = qd.static(1.0e6)
        # Compute contact pairs
        self.n_contact_pairs[None] = 0
        for i_b, i_v in qd.ndrange(self.rigid_solver._B, self.rigid_solver.n_verts):
            if verts_info.is_fixed[i_v]:
                continue
            i_fv = verts_info.verts_state_idx[i_v]
            pos_v = free_verts_state.pos[i_fv, i_b]
            distance = pos_v.z - self.floor_height
            if distance > 0.0:
                continue
            i_g = verts_info.geom_idx[i_v]
            i_l = geoms_info.link_idx[i_g]
            i_p = qd.atomic_add(self.n_contact_pairs[None], 1)
            if i_p < self.max_contact_pairs:
                self.contact_pairs[i_p].batch_idx = i_b
                self.contact_pairs[i_p].link_idx = i_l
                self.contact_pairs[i_p].contact_pos = pos_v
                sap_info[i_p].k = C
                sap_info[i_p].phi0 = distance
                sap_info[i_p].mu = geoms_info.coup_friction[i_g]
            else:
                overflow = True
        return overflow


@qd.data_oriented
class RigidFloorTetContactHandler(RigidContactHandler):
    def __init__(self, simulator: "Simulator", eps: float = 1e-10) -> None:
        super().__init__(simulator)
        self.name = "RigidFloorTetContactHandler"
        self.rigid_solver = self.sim.rigid_solver
        self.floor_height = self.sim.fem_solver.floor_height
        self.eps = eps
        self.contact_candidate_type = qd.types.struct(
            batch_idx=gs.qd_int,  # batch index
            geom_idx=gs.qd_int,  # index of the element
            intersection_code=gs.qd_int,  # intersection code for the element
            distance=gs.qd_vec4,  # distance vector for the element
        )
        self.n_contact_candidates = qd.field(gs.qd_int, shape=())
        self.max_contact_candidates = self.coupler.rigid_volume_elems.shape[0] * self.sim._B * 8
        self.contact_candidates = self.contact_candidate_type.field(shape=(self.max_contact_candidates,))

        self.contact_pair_type = qd.types.struct(
            batch_idx=gs.qd_int,  # batch index
            link_idx=gs.qd_int,  # index of the link
            contact_pos=gs.qd_vec3,  # contact position
            sap_info=self.sap_contact_info_type,  # contact info
        )
        self.max_contact_pairs = self.coupler.rigid_volume_elems.shape[0] * self.sim._B
        self.contact_pairs = self.contact_pair_type.field(shape=(self.max_contact_pairs,))
        self.Jt = qd.field(gs.qd_vec3, shape=(self.max_contact_pairs, self.rigid_solver.n_dofs))
        self.M_inv_Jt = qd.field(gs.qd_vec3, shape=(self.max_contact_pairs, self.rigid_solver.n_dofs))
        self.W = qd.field(gs.qd_mat3, shape=(self.max_contact_pairs,))

    @qd.func
    def detection(
        self,
        f: qd.i32,
        links_info: array_class.LinksInfo,
        verts_info: array_class.VertsInfo,
        faces_info: array_class.FacesInfo,
        free_verts_state: array_class.VertsState,
        fixed_verts_state: array_class.VertsState,
        geoms_info: array_class.GeomsInfo,
    ):
        overflow = False
        candidates = qd.static(self.contact_candidates)
        # Compute contact pairs
        self.n_contact_candidates[None] = 0
        # TODO Check surface element only instead of all elements
        for i_b, i_e in qd.ndrange(self.sim._B, self.coupler.n_rigid_volume_elems):
            i_g = self.coupler.rigid_volume_elems_geom_idx[i_e]
            i_l = geoms_info.link_idx[i_g]
            if links_info.is_fixed[i_l]:
                continue
            intersection_code = qd.int32(0)
            distance = qd.Vector.zero(gs.qd_float, 4)
            for i in qd.static(range(4)):
                i_v = self.coupler.rigid_volume_elems[i_e][i]
                pos_v = self.coupler.rigid_volume_verts[i_b, i_v]
                distance[i] = pos_v.z - self.floor_height
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
                    overflow = True

        pairs = qd.static(self.contact_pairs)
        sap_info = qd.static(pairs.sap_info)
        self.n_contact_pairs[None] = 0
        # Compute pair from candidates
        result_count = qd.min(self.n_contact_candidates[None], self.max_contact_candidates)
        for i_c in range(result_count):
            candidate = candidates[i_c]
            i_b = candidate.batch_idx
            i_e = candidate.geom_idx
            intersection_code = candidate.intersection_code
            distance = candidate.distance
            intersected_edges = self.coupler.MarchingTetsEdgeTable[intersection_code]
            tet_vertices = qd.Matrix.zero(gs.qd_float, 3, 4)  # 4 vertices
            tet_pressures = qd.Vector.zero(gs.qd_float, 4)  # pressures at the vertices

            for i in qd.static(range(4)):
                i_v = self.coupler.rigid_volume_elems[i_e][i]
                tet_vertices[:, i] = self.coupler.rigid_volume_verts[i_b, i_v]
                tet_pressures[i] = self.coupler.rigid_pressure_field[i_v]

            polygon_vertices = qd.Matrix.zero(gs.qd_float, 3, 4)  # 3 or 4 vertices
            total_area = gs.EPS  # avoid division by zero
            total_area_weighted_centroid = qd.Vector([0.0, 0.0, 0.0])
            for i in range(4):
                if intersected_edges[i] >= 0:
                    edge = self.coupler.TetEdges[intersected_edges[i]]
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

            rigid_g = self.coupler.rigid_pressure_gradient[i_b, i_e].z
            g = rigid_g  # harmonic average
            rigid_k = total_area * g
            rigid_phi0 = -pressure / g
            if rigid_k < self.eps or rigid_phi0 > self.eps:
                continue
            i_p = qd.atomic_add(self.n_contact_pairs[None], 1)
            i_g = self.coupler.rigid_volume_elems_geom_idx[i_e]
            i_l = geoms_info.link_idx[i_g]
            if i_p < self.max_contact_pairs:
                pairs[i_p].batch_idx = i_b
                pairs[i_p].link_idx = i_l
                pairs[i_p].contact_pos = centroid
                sap_info[i_p].k = rigid_k
                sap_info[i_p].phi0 = rigid_phi0
                sap_info[i_p].mu = geoms_info.coup_friction[i_g]
            else:
                overflow = True

        return overflow


@qd.data_oriented
class RigidFemTriTetContactHandler(RigidFEMContactHandler):
    """
    Class for handling self-contact between tetrahedral elements in a simulation using hydroelastic model.

    This class extends the FEMContact class and provides methods for detecting self-contact
    between tetrahedral elements, computing contact pairs, and managing contact-related computations.
    """

    def __init__(self, simulator: "Simulator", eps: float = 1e-10) -> None:
        super().__init__(simulator)
        self.name = "RigidFemTriTetContactHandler"
        self.eps = eps
        self.contact_candidate_type = qd.types.struct(
            batch_idx=gs.qd_int,  # batch index
            geom_idx0=gs.qd_int,  # index of the FEM element
            geom_idx1=gs.qd_int,  # index of the Rigid Triangle
            vert_idx1=gs.qd_ivec3,  # vertex indices of the rigid triangle
            normal=gs.qd_vec3,  # contact plane normal
            x=gs.qd_vec3,  # a point on the contact plane
        )
        self.n_contact_candidates = qd.field(gs.qd_int, shape=())
        self.max_contact_candidates = (
            max(self.fem_solver.n_surface_elements, self.rigid_solver.n_faces) * self.fem_solver._B * 8
        )
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
            sap_info=self.sap_contact_info_type,  # contact info
        )
        self.max_contact_pairs = max(self.fem_solver.n_surface_elements, self.rigid_solver.n_faces) * self.fem_solver._B
        self.contact_pairs = self.contact_pair_type.field(shape=(self.max_contact_pairs,))
        self.Jt = qd.field(gs.qd_vec3, shape=(self.max_contact_pairs, self.rigid_solver.n_dofs))
        self.M_inv_Jt = qd.field(gs.qd_vec3, shape=(self.max_contact_pairs, self.rigid_solver.n_dofs))
        self.W = qd.field(gs.qd_mat3, shape=(self.max_contact_pairs,))

    @qd.func
    def compute_candidates(
        self,
        f: qd.i32,
        faces_info: array_class.FacesInfo,
        verts_info: array_class.VertsInfo,
        free_verts_state: array_class.VertsState,
        fixed_verts_state: array_class.VertsState,
    ):
        self.n_contact_candidates[None] = 0
        overflow = False
        result_count = qd.min(
            self.coupler.rigid_tri_bvh.query_result_count[None], self.coupler.rigid_tri_bvh.max_query_results
        )
        for i_r in range(result_count):
            i_b, i_a, i_sq = self.coupler.rigid_tri_bvh.query_result[i_r]
            i_q = self.fem_solver.surface_elements[i_sq]

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
            if magnitude_sqr < gs.EPS:
                continue
            normal *= qd.rsqrt(magnitude_sqr)
            g0 = self.coupler.fem_pressure_gradient[i_b, i_q]
            if g0.dot(normal) < gs.EPS:
                continue

            intersection_code = qd.int32(0)
            for i in qd.static(range(4)):
                i_v = self.fem_solver.elements_i[i_q].el2v[i]
                pos_v = self.fem_solver.elements_v[f, i_v, i_b].pos
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
                overflow = True
        return overflow

    @qd.func
    def compute_pairs(
        self,
        f: qd.i32,
        verts_info: array_class.VertsInfo,
        geoms_info: array_class.GeomsInfo,
        free_verts_state: array_class.VertsState,
        fixed_verts_state: array_class.VertsState,
    ):
        """
        Computes the tet triangle intersection pair and their properties.

        Intersection code reference:
        https://github.com/RobotLocomotion/drake/blob/49ab120ec6f5981484918daa821fc7101e10ebc6/geometry/proximity/mesh_intersection.cc
        """
        sap_info = qd.static(self.contact_pairs.sap_info)
        overflow = False
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
                i_v = self.fem_solver.elements_i[i_e].el2v[i]
                tet_vertices[:, i] = self.fem_solver.elements_v[f, i_v, i_b].pos
                tet_pressures[i] = self.coupler.fem_pressure[i_v]

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
            deformable_g = self.coupler._hydroelastic_stiffness
            rigid_g = self.coupler.fem_pressure_gradient[i_b, i_e].dot(self.contact_candidates[i_c].normal)
            pressure = barycentric0.dot(tet_pressures)
            if total_area < self.eps or rigid_g < self.eps:
                continue
            g = rigid_g * deformable_g / (deformable_g + rigid_g)  # harmonic average
            rigid_k = total_area * g
            rigid_phi0 = -pressure / g
            i_g = verts_info.geom_idx[self.contact_candidates[i_c].vert_idx1[0]]
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
                sap_info[i_p].k = rigid_k
                sap_info[i_p].phi0 = rigid_phi0
                sap_info[i_p].mu = qd.sqrt(self.fem_solver.elements_i[i_e].friction_mu * geoms_info.coup_friction[i_g])
            else:
                overflow = True

        return overflow

    @qd.func
    def detection(
        self,
        f: qd.i32,
        links_info: array_class.LinksInfo,
        verts_info: array_class.VertsInfo,
        faces_info: array_class.FacesInfo,
        free_verts_state: array_class.VertsState,
        fixed_verts_state: array_class.VertsState,
        geoms_info: array_class.GeomsInfo,
    ):
        overflow = False
        overflow |= self.coupler.rigid_tri_bvh.query(self.coupler.fem_surface_tet_aabb.aabbs)
        overflow |= self.compute_candidates(f, faces_info, verts_info, free_verts_state, fixed_verts_state)
        overflow |= self.compute_pairs(f, verts_info, geoms_info, free_verts_state, fixed_verts_state)
        return overflow

    @qd.func
    def compute_delassus_world_frame(self, entities_info: array_class.EntitiesInfo, rigid_info: array_class.RigidInfo):
        dt2_inv = 1.0 / self.sim._substep_dt**2
        # rigid
        self.coupler.rigid_solve_jacobian(
            self.Jt,
            self.M_inv_Jt,
            self.n_contact_pairs[None],
            self.contact_pairs.batch_idx,
            3,
            entities_info=entities_info,
            rigid_info=rigid_info,
        )
        self.W.fill(0.0)
        for i_p, i_d, i, j in qd.ndrange(self.n_contact_pairs[None], self.rigid_solver.n_dofs, 3, 3):
            self.W[i_p][i, j] += self.M_inv_Jt[i_p, i_d][i] * self.Jt[i_p, i_d][j]

        # fem
        barycentric0 = qd.static(self.contact_pairs.barycentric0)
        for i_p in range(self.n_contact_pairs[None]):
            i_g0 = self.contact_pairs[i_p].geom_idx0
            i_b = self.contact_pairs[i_p].batch_idx
            for i in qd.static(range(4)):
                i_v = self.fem_solver.elements_i[i_g0].el2v[i]
                self.W[i_p] += barycentric0[i_p][i] ** 2 * dt2_inv * self.fem_solver.pcg_state_v[i_b, i_v].prec

    @qd.func
    def compute_delassus(self, i_p):
        world = qd.Matrix.cols(
            [self.contact_pairs[i_p].tangent0, self.contact_pairs[i_p].tangent1, self.contact_pairs[i_p].normal]
        )
        return world.transpose() @ self.W[i_p] @ world

    @qd.func
    def compute_Jx(self, i_p, x0, x1):
        """
        Compute the contact Jacobian J times a vector x.
        """
        i_b = self.contact_pairs[i_p].batch_idx
        i_g0 = self.contact_pairs[i_p].geom_idx0
        Jx = qd.Vector.zero(gs.qd_float, 3)

        # fem
        for i in qd.static(range(4)):
            i_v = self.fem_solver.elements_i[i_g0].el2v[i]
            Jx = Jx + self.contact_pairs[i_p].barycentric0[i] * x0[i_b, i_v]

        # rigid
        for i in range(self.rigid_solver.n_dofs):
            Jx = Jx - self.Jt[i_p, i] * x1[i_b, i]
        return qd.Vector(
            [
                Jx.dot(self.contact_pairs[i_p].tangent0),
                Jx.dot(self.contact_pairs[i_p].tangent1),
                Jx.dot(self.contact_pairs[i_p].normal),
            ]
        )

    @qd.func
    def add_Jt_x(self, y0, y1, i_p, x):
        i_b = self.contact_pairs[i_p].batch_idx
        i_g0 = self.contact_pairs[i_p].geom_idx0
        world = qd.Matrix.cols(
            [self.contact_pairs[i_p].tangent0, self.contact_pairs[i_p].tangent1, self.contact_pairs[i_p].normal]
        )
        x_ = world @ x

        # fem
        for i in qd.static(range(4)):
            i_v = self.fem_solver.elements_i[i_g0].el2v[i]
            y0[i_b, i_v] += self.contact_pairs[i_p].barycentric0[i] * x_

        # rigid
        for i in range(self.rigid_solver.n_dofs):
            y1[i_b, i] -= self.Jt[i_p, i].dot(x_)

    @qd.func
    def add_Jt_A_J_diag3x3(self, y, i_p, A):
        i_b = self.contact_pairs[i_p].batch_idx
        i_g0 = self.contact_pairs[i_p].geom_idx0
        world = qd.Matrix.cols(
            [self.contact_pairs[i_p].tangent0, self.contact_pairs[i_p].tangent1, self.contact_pairs[i_p].normal]
        )
        B_ = world @ A @ world.transpose()
        for i in qd.static(range(4)):
            i_v = self.fem_solver.elements_i[i_g0].el2v[i]
            if i_v < self.fem_solver.n_vertices:
                y[i_b, i_v] += self.contact_pairs[i_p].barycentric0[i] ** 2 * B_


@qd.data_oriented
class RigidRigidTetContactHandler(RigidRigidContactHandler):
    """
    Class for handling contact between Rigid bodies using hydroelastic model.

    This class extends the RigidContact class and provides methods for detecting contact
    between tetrahedral elements, computing contact pairs, and managing contact-related computations.
    """

    def __init__(self, simulator: "Simulator", eps: float = 1e-10) -> None:
        super().__init__(simulator)
        self.coupler = simulator.coupler
        self.name = "RigidRigidTetContactHandler"
        self.eps = eps
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
        self.max_contact_candidates = self.coupler.rigid_volume_elems.shape[0] * self.sim._B * 8
        self.contact_candidates = self.contact_candidate_type.field(shape=(self.max_contact_candidates,))

        self.contact_pair_type = qd.types.struct(
            batch_idx=gs.qd_int,  # batch index
            normal=gs.qd_vec3,  # contact plane normal
            tangent0=gs.qd_vec3,  # contact plane tangent0
            tangent1=gs.qd_vec3,  # contact plane tangent1
            link_idx0=gs.qd_int,  # index of the link
            link_idx1=gs.qd_int,  # index of the other link
            contact_pos=gs.qd_vec3,  # contact position
            sap_info=self.sap_contact_info_type,  # contact info
        )
        self.max_contact_pairs = self.coupler.rigid_volume_elems.shape[0] * self.sim._B
        self.contact_pairs = self.contact_pair_type.field(shape=(self.max_contact_pairs,))
        self.Jt = qd.field(gs.qd_vec3, shape=(self.max_contact_pairs, self.rigid_solver.n_dofs))
        self.M_inv_Jt = qd.field(gs.qd_vec3, shape=(self.max_contact_pairs, self.rigid_solver.n_dofs))
        self.W = qd.field(gs.qd_mat3, shape=(self.max_contact_pairs,))

    @qd.func
    def compute_candidates(self, f: qd.i32):
        overflow = False
        candidates = qd.static(self.contact_candidates)
        self.n_contact_candidates[None] = 0
        result_count = qd.min(
            self.coupler.rigid_tet_bvh.query_result_count[None], self.coupler.rigid_tet_bvh.max_query_results
        )
        for i_r in range(result_count):
            i_b, i_a, i_q = self.coupler.rigid_tet_bvh.query_result[i_r]
            i_v0 = self.coupler.rigid_volume_elems[i_a][0]
            i_v1 = self.coupler.rigid_volume_elems[i_q][1]
            x0 = self.coupler.rigid_volume_verts[i_b, i_v0]
            x1 = self.coupler.rigid_volume_verts[i_b, i_v1]
            p0 = self.coupler.rigid_pressure_field[i_v0]
            p1 = self.coupler.rigid_pressure_field[i_v1]
            g0 = self.coupler.rigid_pressure_gradient[i_b, i_a]
            g1 = self.coupler.rigid_pressure_gradient[i_b, i_q]
            g0_norm = g0.norm()
            g1_norm = g1.norm()
            if g0_norm < gs.EPS or g1_norm < gs.EPS:
                continue
            # Calculate the isosurface, i.e. equal pressure plane defined by x and normal
            # Solve for p0 + g0.dot(x - x0) = p1 + g1.dot(x - x1)
            normal = g0 - g1
            magnitude = normal.norm()
            if magnitude < gs.EPS:
                continue
            normal /= magnitude
            b = p1 - p0 - g1.dot(x1) + g0.dot(x0)
            x = b / magnitude * normal
            # Check that the normal is pointing along g0 and against g1, some allowance as used in Drake
            if normal.dot(g0) < self.eps or normal.dot(g1) > -self.eps:
                continue

            intersection_code0 = qd.int32(0)
            distance0 = qd.Vector([0.0, 0.0, 0.0, 0.0])
            intersection_code1 = qd.int32(0)
            distance1 = qd.Vector([0.0, 0.0, 0.0, 0.0])
            for i in qd.static(range(4)):
                i_v = self.coupler.rigid_volume_elems[i_a][i]
                pos_v = self.coupler.rigid_volume_verts[i_b, i_v]
                distance0[i] = (pos_v - x).dot(normal)  # signed distance
                if distance0[i] > 0:
                    intersection_code0 |= 1 << i
            for i in qd.static(range(4)):
                i_v = self.coupler.rigid_volume_elems[i_q][i]
                pos_v = self.coupler.rigid_volume_verts[i_b, i_v]
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
                overflow = True
        return overflow

    @qd.func
    def compute_pairs(self, i_step: qd.i32, geoms_info: array_class.GeomsInfo):
        overflow = False
        candidates = qd.static(self.contact_candidates)
        pairs = qd.static(self.contact_pairs)
        sap_info = qd.static(pairs.sap_info)
        normal_signs = qd.Vector([1.0, -1.0, 1.0, -1.0])  # make normal point outward
        self.n_contact_pairs[None] = 0
        result_count = qd.min(self.n_contact_candidates[None], self.max_contact_candidates)
        for i_c in range(result_count):
            i_b = candidates[i_c].batch_idx
            i_e0 = candidates[i_c].geom_idx0
            i_e1 = candidates[i_c].geom_idx1
            intersection_code0 = candidates[i_c].intersection_code0
            distance0 = candidates[i_c].distance0
            intersected_edges0 = self.coupler.MarchingTetsEdgeTable[intersection_code0]
            tet_vertices0 = qd.Matrix.zero(gs.qd_float, 3, 4)  # 4 vertices of tet 0
            tet_pressures0 = qd.Vector.zero(gs.qd_float, 4)  # pressures at the vertices of tet 0
            tet_vertices1 = qd.Matrix.zero(gs.qd_float, 3, 4)  # 4 vertices of tet 1

            for i in qd.static(range(4)):
                i_v = self.coupler.rigid_volume_elems[i_e0][i]
                tet_vertices0[:, i] = self.coupler.rigid_volume_verts[i_b, i_v]
                tet_pressures0[i] = self.coupler.rigid_pressure_field[i_v]

            for i in qd.static(range(4)):
                i_v = self.coupler.rigid_volume_elems[i_e1][i]
                tet_vertices1[:, i] = self.coupler.rigid_volume_verts[i_b, i_v]

            polygon_vertices = qd.Matrix.zero(gs.qd_float, 3, 8)  # maximum 8 vertices
            polygon_n_vertices = gs.qd_int(0)
            clipped_vertices = qd.Matrix.zero(gs.qd_float, 3, 8)  # maximum 8 vertices
            clipped_n_vertices = gs.qd_int(0)
            for i in range(4):
                if intersected_edges0[i] >= 0:
                    edge = self.coupler.TetEdges[intersected_edges0[i]]
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

            if total_area < self.eps:
                continue
            centroid = total_area_weighted_centroid / total_area
            tangent0 = polygon_vertices[:, 0] - centroid
            tangent0 /= tangent0.norm()
            tangent1 = candidates[i_c].normal.cross(tangent0)
            g0 = self.coupler.rigid_pressure_gradient[i_b, i_e0].dot(candidates[i_c].normal)
            g1 = -self.coupler.rigid_pressure_gradient[i_b, i_e1].dot(candidates[i_c].normal)
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
            if rigid_phi0 > self.eps:
                continue
            i_p = qd.atomic_add(self.n_contact_pairs[None], 1)
            if i_p < self.max_contact_pairs:
                pairs[i_p].batch_idx = i_b
                pairs[i_p].normal = candidates[i_c].normal
                pairs[i_p].tangent0 = tangent0
                pairs[i_p].tangent1 = tangent1
                pairs[i_p].contact_pos = centroid
                i_g0 = self.coupler.rigid_volume_elems_geom_idx[i_e0]
                i_g1 = self.coupler.rigid_volume_elems_geom_idx[i_e1]
                i_l0 = geoms_info.link_idx[i_g0]
                i_l1 = geoms_info.link_idx[i_g1]
                pairs[i_p].link_idx0 = i_l0
                pairs[i_p].link_idx1 = i_l1
                sap_info[i_p].k = rigid_k
                sap_info[i_p].phi0 = rigid_phi0
                sap_info[i_p].mu = qd.sqrt(geoms_info.friction[i_g0] * geoms_info.friction[i_g1])
            else:
                overflow = True
        return overflow

    @qd.func
    def detection(
        self,
        f: qd.i32,
        links_info: array_class.LinksInfo,
        verts_info: array_class.VertsInfo,
        faces_info: array_class.FacesInfo,
        free_verts_state: array_class.VertsState,
        fixed_verts_state: array_class.VertsState,
        geoms_info: array_class.GeomsInfo,
    ):
        overflow = False
        overflow |= self.coupler.rigid_tet_bvh.query(self.coupler.rigid_tet_aabb.aabbs)
        overflow |= self.compute_candidates(f)
        overflow |= self.compute_pairs(f, geoms_info)
        return overflow
