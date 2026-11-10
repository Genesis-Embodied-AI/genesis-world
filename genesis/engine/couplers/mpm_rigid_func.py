"""Contact pairs between the implicit material point method (MPM) and Rigid in the joint Newton solve."""

import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
import genesis.utils.sdf as sdf
from genesis.engine.solvers.mpm_data import PYRAMID_ROWS, MpmInfo
from genesis.engine.solvers.mpm_func import (
    band_row,
    cofactor,
    cofactor_jacobian,
    contact_band_width,
    node_offset,
    pyramid_directions,
)
from genesis.engine.solvers.newton_linear import PAIRS
from genesis.engine.solvers.rigid.abd.forward_dynamics import func_vel_at_point
from genesis.engine.solvers.rigid.newton_participant import RigidNewtonParticipant
from genesis.utils.array_class import GeomsInfo, GeomsState, RigidInfo, SDFInfo
from genesis.utils.data import Data, DataKind, Size, field

from .contact_candidates import LeafTrees, empty_box

COUPLED_GEOMS = Size("coupled_geoms")


class CoupledGeoms(Data, kind=DataKind.CONSTANT):
    """The Rigid geoms coupled with the material point method (MPM), one leaf each in the tree of RigidGeomQuery."""

    idx: qd.Tensor = field(int, shape=(COUPLED_GEOMS,), ndarray=True)


class MpmRigidPairs(Data, kind=DataKind.SCRATCH):
    """The pairs of the particles with their nearest coupled Rigid geom (see MpmRigidContact.build_pair)."""

    n_pairs: qd.Tensor = field(int, ndarray=True)
    sorted_idx: qd.Tensor = field(int, shape=(PAIRS,), ndarray=True)
    i_b: qd.Tensor = field(int, shape=(PAIRS,), ndarray=True)
    geom_idx: qd.Tensor = field(int, shape=(PAIRS,), ndarray=True)
    link_idx: qd.Tensor = field(int, shape=(PAIRS,), ndarray=True)
    phi: qd.Tensor = field(float, shape=(PAIRS,), ndarray=True)
    normal: qd.Tensor = field((float, 3), shape=(PAIRS,), ndarray=True)
    friction: qd.Tensor = field(float, shape=(PAIRS,), ndarray=True)
    stiffness: qd.Tensor = field((float, PYRAMID_ROWS), shape=(PAIRS,), ndarray=True)
    target: qd.Tensor = field((float, PYRAMID_ROWS), shape=(PAIRS,), ndarray=True)


# ============================================ Layer 0: spatial algebra =============================================


@qd.func
def contact_arm(x: qd.types.vector(3), root_com: qd.types.vector(3)):
    """Arm of a point from the root center of mass (COM) of a link, about which the link moves and takes torques."""
    return x - root_com


@qd.func
def point_motion(lin: qd.types.vector(3), ang: qd.types.vector(3), r: qd.types.vector(3)):
    """Displacement of a point at arm r of a link from the spatial displacement (lin, ang) of the link."""
    return lin + ang.cross(r)


@qd.func
def point_wrench(y: qd.types.vector(3), r: qd.types.vector(3)):
    """Spatial force on a link of a force y on a point at arm r of it."""
    return y, r.cross(y)


@qd.func
def point_stiffness(H: qd.types.matrix(3, 3), r: qd.types.vector(3)):
    """Spatial stiffness P^T H P of a link from a stiffness H on a point at arm r of it, P = [I, -[r]x] mapping the
    spatial displacement of the link to the displacement of the point."""
    P = qd.Matrix(
        [
            [1.0, 0.0, 0.0, 0.0, r[2], -r[1]],
            [0.0, 1.0, 0.0, -r[2], 0.0, r[0]],
            [0.0, 0.0, 1.0, r[1], -r[0], 0.0],
        ],
        dt=gs.qd_float,
    )
    return P.transpose() @ H @ P


@qd.func
def posed_box(corners: qd.types.matrix(8, 3), pos: qd.types.vector(3), quat: qd.types.vector(4)):
    """World box of a box posed by a translation and a rotation, from its 8 corners, with its center and the radius of
    its bounding sphere."""
    box_min = gu.qd_vec3(qd.math.inf)
    box_max = gu.qd_vec3(-qd.math.inf)
    for k in qd.static(range(8)):
        corner = gu.qd_transform_by_trans_quat(corners[k, :], pos, quat)
        box_min = qd.min(box_min, corner)
        box_max = qd.max(box_max, corner)
    return box_min, box_max, 0.5 * (box_min + box_max), 0.5 * (box_max - box_min).norm()


@qd.func
def swept_geom_box(
    box_min: qd.types.vector(3),
    box_max: qd.types.vector(3),
    radius: float,
    v_center: qd.types.vector(3),
    omega: qd.types.vector(3),
    softness: float,
    dt: float,
):
    """Box of a geom grown by its contact softness and twice its substep travel at the speed bound of the points within
    the radius of its bounding sphere."""
    margin = contact_band_width(v_center.norm() + omega.norm() * radius, dt, qd.max(softness, 0.0))
    return box_min - margin, box_max + margin


# =============================================== Layer 0: geom distance ===============================================


@qd.func
def sdf_grid_cell(pos_sdf: qd.types.vector(3), sdf_res):
    """Point of a signed distance field (SDF) grid clamped into it, and the lower corner of the cell holding it."""
    upper = (sdf_res - 1).cast(gs.qd_float)
    q = qd.min(qd.max(pos_sdf, 0.0), upper)
    return q, qd.max(qd.min(qd.floor(q, gs.qd_int), sdf_res - 2), 0)


@qd.func
def sdf_grid_terms(
    pos_sdf: qd.types.vector(3),
    q: qd.types.vector(3),
    base,
    corner_values: qd.types.vector(8),
    cell_size: float,
    sdf_res,
    is_hessian: qd.template(),
):
    """Signed distance, gradient and, with `is_hessian`, Hessian in the mesh frame of the trilinear interpolant of a
    signed distance field (SDF) grid at a point in grid coordinates, the values of the corners of its cell in the order
    of node_offset, continued outside the grid by the distance to it."""
    upper = (sdf_res - 1).cast(gs.qd_float)
    value = gs.qd_float(0.0)
    grad = qd.Vector.zero(gs.qd_float, 3)
    hess = qd.Matrix.zero(gs.qd_float, 3, 3)
    for a in qd.static(range(8)):
        offset = node_offset(a)
        w = 1.0 - qd.abs(q - (base + offset).cast(gs.qd_float))
        axis_dw = (2 * offset - 1).cast(gs.qd_float) / cell_size
        value += w[0] * w[1] * w[2] * corner_values[a]
        grad += corner_values[a] * axis_dw * cofactor(w)
        if qd.static(is_hessian):
            hess += corner_values[a] * axis_dw.outer_product(axis_dw) * cofactor_jacobian(w)
    outside = (pos_sdf - q) * cell_size
    distance = outside.norm()
    if distance > 0.0:
        outside_dir = outside / distance
        is_outside = ((pos_sdf < 0.0) | (pos_sdf > upper)).cast(gs.qd_float)
        grad = qd.select(is_outside > 0.0, outside_dir, grad)
        if qd.static(is_hessian):
            hess *= (1.0 - is_outside).outer_product(1.0 - is_outside)
            hess += (
                is_outside.outer_product(is_outside)
                * (qd.Matrix.identity(gs.qd_float, 3) - outside_dir.outer_product(outside_dir))
                / distance
            )
    return value + distance, grad, hess


@qd.func
def world_normal(grad_mesh: qd.types.vector(3), quat: qd.types.vector(4), eps: float):
    """Outward unit normal of a geom from the gradient of its distance in the mesh frame."""
    return gu.qd_normalize(gu.qd_transform_by_quat(grad_mesh, quat), eps)


@qd.func
def geom_contact_reverse(
    is_sphere: bool,
    x: qd.types.vector(3),
    pos: qd.types.vector(3),
    quat: qd.types.vector(4),
    grad_mesh: qd.types.vector(3),
    hess_mesh: qd.types.matrix(3, 3),
    phi_bar: float,
    n_bar: qd.types.vector(3),
    eps: float,
    x_bar: qd.types.vector(3),
):
    """Reverse of the signed distance and of the normal of a geom at a world point, from the gradient and Hessian of
    its distance in the mesh frame there, adding to the adjoint of the point, with the adjoints of the pose of the
    geom."""
    x_bar_geom = qd.Vector.zero(gs.qd_float, 3)
    quat_bar = qd.Vector.zero(gs.qd_float, 4)
    if is_sphere:
        e = x - pos
        e_norm = qd.max(e.norm(), eps)
        n = e / e_norm
        x_bar_geom = phi_bar * n + (n_bar - n.dot(n_bar) * n) / e_norm
    else:
        grad_world = gu.qd_transform_by_quat(grad_mesh, quat)
        grad_world_norm = qd.max(grad_world.norm(), eps)
        n = grad_world / grad_world_norm
        grad_world_bar = (n_bar - n.dot(n_bar) * n) / grad_world_norm
        quat_bar += gu.qd_transform_by_quat_grad_quat(grad_mesh, quat, grad_world_bar)
        quat_inv = gu.qd_inv_quat(quat)
        grad_mesh_bar = gu.qd_transform_by_quat(grad_world_bar, quat_inv)
        # pos_mesh = R^T (x - pos)
        pos_mesh_bar = phi_bar * grad_mesh + hess_mesh @ grad_mesh_bar
        x_bar_geom = gu.qd_transform_by_quat(pos_mesh_bar, quat)
        quat_inv_bar = gu.qd_transform_by_quat_grad_quat(x - pos, quat_inv, pos_mesh_bar)
        quat_bar += gu.qd_inv_quat(quat_inv_bar)
    return x_bar + x_bar_geom, -x_bar_geom, quat_bar


# =============================================== Layer 0: contact rows ===============================================


@qd.func
def rigid_row_coefficients(
    friction: float,
    impratio: float,
    invweight_particle: float,
    invweight_link: float,
    sol_params: qd.types.vector(7),
    dt: float,
):
    """Diagonal coefficient of the rows of a pair of a particle with a Rigid geom, from the inverse weights of both and
    Rigid's regularized friction, with the scale of the inverse weight in it and Rigid's solref / solimp parameters of
    the rows, the time constant floored at twice the substep."""
    friction_sq_reg = friction * friction / impratio
    diag_coef_scale = (1.0 + friction_sq_reg) * 2.0 * friction_sq_reg
    sol_params_row = sol_params
    sol_params_row[0] = qd.max(sol_params[0], 2.0 * dt)
    return (invweight_particle + invweight_link) * diag_coef_scale, diag_coef_scale, sol_params_row


@qd.func
def rigid_pair_rows(
    normal: qd.types.vector(3),
    friction: float,
    diag_coef: float,
    sol_params_row: qd.types.vector(7),
    phi: float,
    v_p: qd.types.vector(3),
    v_r: qd.types.vector(3),
    dt: float,
):
    """Stiffness and target of the rows of the friction pyramid of a pair of a particle with a Rigid geom (see
    rigid_row_coefficients), the target relative to the motion of the contact point over the substep."""
    directions = pyramid_directions(normal, friction)
    v_rel = v_p - v_r
    stiffness = qd.Vector.zero(gs.qd_float, PYRAMID_ROWS)
    target = qd.Vector.zero(gs.qd_float, PYRAMID_ROWS)
    for k in range(PYRAMID_ROWS):
        direction = qd.Vector([directions[k, j] for j in qd.static(range(3))], dt=gs.qd_float)
        stiffness_k, target_k = band_row(sol_params_row, phi, direction.dot(v_rel), diag_coef, dt)
        stiffness[k] = stiffness_k
        target[k] = target_k + dt * direction.dot(v_r)
    return stiffness, target


# ============================================ Layer 1: geom distance =============================================


@qd.func
def geom_sdf_grid(
    i_g: int, i_b: int, x: qd.types.vector(3), geoms_state: GeomsState, sdf_info: SDFInfo, is_hessian: qd.template()
):
    """Signed distance, gradient and Hessian in the mesh frame of a geom with a signed distance field (SDF) grid at a
    world point (see sdf_grid_terms)."""
    pos_mesh = gu.qd_inv_transform_by_trans_quat(x, geoms_state.pos[i_g, i_b], geoms_state.quat[i_g, i_b])
    pos_sdf = gu.qd_transform_by_T(pos_mesh, sdf_info.geoms_info.T_mesh_to_sdf[i_g])
    sdf_res = sdf_info.geoms_info.sdf_res[i_g]
    q, base = sdf_grid_cell(pos_sdf, sdf_res)
    corner_values = qd.Vector(
        [
            sdf_info.geoms_sdf_val[sdf.sdf_func_ravel_cell_idx(base + node_offset(a), i_g, sdf_res, sdf_info)]
            for a in qd.static(range(8))
        ],
        dt=gs.qd_float,
    )
    return sdf_grid_terms(pos_sdf, q, base, corner_values, sdf_info.geoms_info.sdf_cell_size[i_g], sdf_res, is_hessian)


@qd.func
def geom_sdf(
    i_g: int, i_b: int, x: qd.types.vector(3), geoms_state: GeomsState, geoms_info: GeomsInfo, sdf_info: SDFInfo
):
    """Signed distance of a world point to a geom, by `geom_sdf_grid` for the geoms with a distance grid."""
    signed_distance = gs.qd_float(0.0)
    geom_type = geoms_info.type[i_g]
    if geom_type == gs.GEOM_TYPE.SPHERE or geom_type == gs.GEOM_TYPE.PLANE or geom_type == gs.GEOM_TYPE.TERRAIN:
        signed_distance = sdf.sdf_func_world(i_g, i_b, x, geoms_state, geoms_info, sdf_info)
    else:
        signed_distance, grad_mesh, hess_mesh = geom_sdf_grid(i_g, i_b, x, geoms_state, sdf_info, is_hessian=False)
    return signed_distance


@qd.func
def geom_normal(
    i_g: int,
    i_b: int,
    x: qd.types.vector(3),
    geoms_state: GeomsState,
    geoms_info: GeomsInfo,
    rigid_info: RigidInfo,
    sdf_info: SDFInfo,
    collider_config: qd.template(),
):
    """Outward unit normal of a geom at a world point: the gradient of the distance measured by `geom_sdf`."""
    normal = qd.Vector.zero(gs.qd_float, 3)
    geom_type = geoms_info.type[i_g]
    if geom_type == gs.GEOM_TYPE.SPHERE or geom_type == gs.GEOM_TYPE.PLANE or geom_type == gs.GEOM_TYPE.TERRAIN:
        normal = sdf.sdf_func_normal_world(i_g, i_b, x, geoms_state, geoms_info, rigid_info, sdf_info, collider_config)
    else:
        signed_distance, grad_mesh, hess_mesh = geom_sdf_grid(i_g, i_b, x, geoms_state, sdf_info, is_hessian=False)
        normal = world_normal(grad_mesh, geoms_state.quat[i_g, i_b], rigid_info.EPS[None])
    return normal


# ============================================== Layer 1: broad phase ==============================================


@qd.data_oriented
class RigidGeomQuery(LeafTrees):
    """Linear bounding volume hierarchy (LBVH) over the coupled Rigid geoms of every environment."""

    def __init__(self, rigid: RigidNewtonParticipant, n_envs: int):
        self.n_coupled = len(rigid.coupled_geoms_idx)
        self.coupled_geoms = CoupledGeoms(sizes={COUPLED_GEOMS: self.n_coupled})
        if self.n_coupled:
            self.coupled_geoms.idx.from_numpy(rigid.coupled_geoms_idx)
        super().__init__(n_envs, self.n_coupled, is_active=self.n_coupled > 0)
        self.geoms_state = rigid.dyn_state.geoms
        self.geoms_info = rigid.dyn_info.geoms
        self.geoms_init_AABB = rigid.geoms_init_AABB
        self.links_state = rigid.dyn_state.links
        self.links_info = rigid.dyn_info.links
        self.batch_links_info = rigid.config.rigid_batch_links_info

    def build(self, mpm_info):
        """Fit the leaf boxes to the geoms of the substep and build the trees."""
        if self.n_coupled:
            kernel_fit_geom_leaves(self, mpm_info)
            self.build_trees()


@qd.kernel
def kernel_fit_geom_leaves(query: qd.template(), mpm_info: MpmInfo):
    """Fit the leaf box of every coupled geom to the world box of its posed initial box."""
    for i_b, i_a in qd.ndrange(query.leaves.aabbs_min.shape[0], query.leaves.aabbs_min.shape[1]):
        i_g = query.coupled_geoms.idx[i_a]
        i_l = query.geoms_info.link_idx[i_g]
        is_carried = True
        if qd.static(query.batch_links_info):
            is_carried = query.links_info.geom_start[i_l, i_b] <= i_g < query.links_info.geom_end[i_l, i_b]
        lo, hi = empty_box()
        if is_carried:
            box_min, box_max, center, radius = posed_box(
                qd.Matrix.rows([query.geoms_init_AABB[i_g, i_corner] for i_corner in qd.static(range(8))]),
                query.geoms_state.pos[i_g, i_b],
                query.geoms_state.quat[i_g, i_b],
            )
            lo, hi = swept_geom_box(
                box_min,
                box_max,
                radius,
                func_vel_at_point(i_l, i_b, center, query.links_state),
                query.links_state.cd_ang[i_l, i_b],
                query.geoms_info.coup_softness[i_g],
                mpm_info.substep_dt[()],
            )
        query.leaves.aabbs_min[i_b, i_a] = lo
        query.leaves.aabbs_max[i_b, i_a] = hi
