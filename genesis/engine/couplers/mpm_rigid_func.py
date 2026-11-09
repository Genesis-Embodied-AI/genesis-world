"""Contact pairs between the implicit material point method (MPM) and Rigid in the joint Newton solve."""

import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
import genesis.utils.sdf as sdf
from genesis.engine.bvh import build_bvh, get_bvh_data
from genesis.engine.solvers.mpm_data import PYRAMID_ROWS, MpmInfo
from genesis.engine.solvers.mpm_func import EMPTY_BOX_HALF, cofactor
from genesis.engine.solvers.newton_linear import PAIRS
from genesis.engine.solvers.rigid.abd.forward_dynamics import func_vel_at_point
from genesis.engine.solvers.rigid.newton_participant import RigidNewtonParticipant
from genesis.utils.array_class import GeomsInfo, GeomsState, RigidInfo, SDFInfo
from genesis.utils.data import Data, DataKind, Size, field

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


# =============================================== Layer 0: geom distance ===============================================


@qd.func
def geom_sdf_grid(i_g: int, i_b: int, x: qd.types.vector(3), geoms_state: GeomsState, sdf_info: SDFInfo):
    """Continuous signed distance and its gradient in the mesh frame of a geom with a signed distance field (SDF)
    grid."""
    pos_mesh = gu.qd_inv_transform_by_trans_quat(x, geoms_state.pos[i_g, i_b], geoms_state.quat[i_g, i_b])
    pos_sdf = gu.qd_transform_by_T(pos_mesh, sdf_info.geoms_info.T_mesh_to_sdf[i_g])
    sdf_res = sdf_info.geoms_info.sdf_res[i_g]
    cell_size = sdf_info.geoms_info.sdf_cell_size[i_g]
    upper = (sdf_res - 1).cast(gs.qd_float)
    q = qd.min(qd.max(pos_sdf, 0.0), upper)
    base = qd.max(qd.min(qd.floor(q, gs.qd_int), sdf_res - 2), 0)
    value = gs.qd_float(0.0)
    grad = qd.Vector.zero(gs.qd_float, 3)
    for offset in qd.grouped(qd.ndrange(2, 2, 2)):
        node = base + offset
        w = 1.0 - qd.abs(q - node.cast(gs.qd_float))
        sdf_val = sdf_info.geoms_sdf_val[sdf.sdf_func_ravel_cell_idx(node, i_g, sdf_res, sdf_info)]
        value += w[0] * w[1] * w[2] * sdf_val
        grad += sdf_val * (2 * offset - 1).cast(gs.qd_float) * cofactor(w) / cell_size
    outside = (pos_sdf - q) * cell_size
    distance = outside.norm()
    if distance > 0.0:
        grad = qd.select((pos_sdf < 0.0) | (pos_sdf > upper), outside / distance, grad)
    return value + distance, grad


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
        signed_distance, _ = geom_sdf_grid(i_g, i_b, x, geoms_state, sdf_info)
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
        _, grad_mesh = geom_sdf_grid(i_g, i_b, x, geoms_state, sdf_info)
        normal = gu.qd_normalize(gu.qd_transform_by_quat(grad_mesh, geoms_state.quat[i_g, i_b]), rigid_info.EPS[None])
    return normal


# ============================================== Layer 1: broad phase ==============================================


@qd.data_oriented
class RigidGeomQuery:
    """Linear bounding volume hierarchy (LBVH) over the coupled Rigid geoms of every environment."""

    def __init__(self, rigid: RigidNewtonParticipant, n_envs: int):
        self.n_coupled = len(rigid.coupled_geoms_idx)
        self.coupled_geoms = CoupledGeoms(sizes={COUPLED_GEOMS: self.n_coupled})
        if self.n_coupled:
            self.coupled_geoms.idx.from_numpy(rigid.coupled_geoms_idx)
        self.bvh_state, self.bvh_config = get_bvh_data(n_envs, max(self.n_coupled, 1), is_active=self.n_coupled > 0)
        self.tree = self.bvh_state.tree
        self.leaves = self.bvh_state.leaves
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
            build_bvh(self.bvh_state, self.bvh_config, eps=gs.EPS)


@qd.kernel
def kernel_fit_geom_leaves(query: qd.template(), mpm_info: MpmInfo):
    """Fit the leaf box of every coupled geom to the world box of its posed initial box.

    The box grows by the contact margin of the geom and by twice the distance its points may travel over the substep. A
    geom that an environment does not carry gets an empty box with a finite center."""
    dt = mpm_info.substep_dt[()]
    for i_b, i_a in qd.ndrange(query.leaves.aabbs_min.shape[0], query.leaves.aabbs_min.shape[1]):
        i_g = query.coupled_geoms.idx[i_a]
        i_l = query.geoms_info.link_idx[i_g]
        is_carried = True
        if qd.static(query.batch_links_info):
            is_carried = query.links_info.geom_start[i_l, i_b] <= i_g < query.links_info.geom_end[i_l, i_b]
        if is_carried:
            geom_pos = query.geoms_state.pos[i_g, i_b]
            geom_quat = query.geoms_state.quat[i_g, i_b]
            box_min = gu.qd_vec3(qd.math.inf)
            box_max = gu.qd_vec3(-qd.math.inf)
            for i_corner in qd.static(range(8)):
                corner = gu.qd_transform_by_trans_quat(query.geoms_init_AABB[i_g, i_corner], geom_pos, geom_quat)
                box_min = qd.min(box_min, corner)
                box_max = qd.max(box_max, corner)
            center = 0.5 * (box_min + box_max)
            radius = 0.5 * (box_max - box_min).norm()
            speed = func_vel_at_point(i_l, i_b, center, query.links_state).norm()
            speed += query.links_state.cd_ang[i_l, i_b].norm() * radius
            margin = qd.max(query.geoms_info.coup_softness[i_g], 0.0) + 2.0 * dt * speed
            query.leaves.aabbs_min[i_b, i_a] = box_min - margin
            query.leaves.aabbs_max[i_b, i_a] = box_max + margin
        else:
            query.leaves.aabbs_min[i_b, i_a] = qd.Vector([EMPTY_BOX_HALF] * 3, dt=gs.qd_float)
            query.leaves.aabbs_max[i_b, i_a] = qd.Vector([-EMPTY_BOX_HALF] * 3, dt=gs.qd_float)
