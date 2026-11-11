from typing import TYPE_CHECKING

import numpy as np

import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
from genesis.engine.bvh import (
    BVHState,
    InstancesQuery,
    PackedInstances,
    build_packed_instances,
    func_build_bvh,
    func_bvh_traverse,
    get_bvh_data,
    kernel_build_instances_tree,
)
from genesis.engine.solvers.rigid.data import DynInfo, DynState
from genesis.engine.state_change import StateChange, Subscriber
from genesis.utils.cache import OwnedCache
from genesis.utils.data import ENVS, Data, DataKind, Size, field
from genesis.utils.misc import qd_to_numpy, tensor_to_array

if TYPE_CHECKING:
    from genesis.engine.solvers.kinematic_solver import KinematicSolver


class RaycastResult(Data, kind=DataKind.SCRATCH, shape=(ENVS,)):
    """
    The nearest hit of a ray cast against the rigid geoms in each environment: its distance, geom, point and normal.
    """

    distance: qd.Tensor = field(float)
    geom_idx: qd.Tensor = field(int)
    hit_point: qd.Tensor = field((float, 3))
    normal: qd.Tensor = field((float, 3))


# Faces of the visual geoms of a solver whose vertices are set per environment
WORLD_FACES = Size("world_faces")


class WorldFaces(Data, kind=DataKind.CONSTANT, shape=(WORLD_FACES,)):
    """The faces of the visual geoms of a solver whose vertices are set per environment (see set_vverts)."""

    # Vertices of each face, as their slots of the per-environment vertices
    faces_verts_idx: qd.Tensor = field((int, 3))
    faces_geom_idx: qd.Tensor = field(int)


# Sensors and geoms of the closest-point queries a mask of geoms restricts (see CandidateGeoms)
CANDIDATE_SENSORS = Size("candidate_sensors")
CANDIDATE_GEOMS = Size("candidate_geoms")


class CandidateGeoms(Data, kind=DataKind.SCRATCH, shape=(CANDIDATE_SENSORS, CANDIDATE_GEOMS, ENVS)):
    """The geoms of a solver that the closest-point queries of each sensor may reach in each environment, or in every
    environment for a single one."""

    is_candidate: qd.Tensor = field(bool)


# Closest-point queries of a sensor class, each of one query point (see ClosestQuery)
CLOSEST_QUERIES = Size("closest_queries")


class ClosestQueries(Data, kind=DataKind.CONSTANT, shape=(CLOSEST_QUERIES,)):
    """The sensor of each closest-point query, which selects the geoms it may reach (see CandidateGeoms)."""

    sensor_idx: qd.Tensor = field(int)


class ClosestHits(Data, kind=DataKind.SCRATCH, shape=(CLOSEST_QUERIES, ENVS)):
    """The closest triangle of each closest-point query: its signed distance, geom and outward normal in the world."""

    signed_dist: qd.Tensor = field(float)
    geom_idx: qd.Tensor = field(int)
    normal: qd.Tensor = field((float, 3))


_SOLVER_MESHES_CACHE = OwnedCache()


def get_solver_meshes(
    solver: "KinematicSolver", is_visual: bool, faces_is_used: np.ndarray | None = None
) -> PackedInstances:
    """Return the trees of the faces of the geoms of a solver, or of its visual geoms, in the frame of each geom.

    The geoms of one mesh share its tree. A geom holding a face that `faces_is_used` drops gets no tree. The trees are
    built once and shared by every user of the solver.
    """

    def compute():
        dyn_info = solver.dyn_info
        if is_visual:
            verts = qd_to_numpy(dyn_info.vverts.init_pos, transpose=True)
            faces_verts_idx = qd_to_numpy(dyn_info.vfaces.vverts_idx, transpose=True)
            geoms_face_start = qd_to_numpy(dyn_info.vgeoms.vface_start, transpose=True)
            geoms_face_end = qd_to_numpy(dyn_info.vgeoms.vface_end, transpose=True)
        else:
            verts = qd_to_numpy(dyn_info.verts.init_pos, transpose=True)
            faces_verts_idx = qd_to_numpy(dyn_info.faces.verts_idx, transpose=True)
            geoms_face_start = qd_to_numpy(dyn_info.geoms.face_start, transpose=True)
            geoms_face_end = qd_to_numpy(dyn_info.geoms.face_end, transpose=True)
        geoms_aabb_min, geoms_aabb_max = [], []
        for i_f_start, i_f_end in zip(geoms_face_start.tolist(), geoms_face_end.tolist()):
            tris = verts[faces_verts_idx[i_f_start:i_f_end]]
            if faces_is_used is not None and not faces_is_used[i_f_start:i_f_end].all():
                tris = tris[:0]
            geoms_aabb_min.append(tris.min(axis=1))
            geoms_aabb_max.append(tris.max(axis=1))
        return build_packed_instances(geoms_face_start.tolist(), geoms_aabb_min, geoms_aabb_max)

    return _SOLVER_MESHES_CACHE.get(solver, (solver, is_visual), compute)


@qd.func
def ray_projection(ray_dir: qd.types.vector(3), eps: float):
    """
    Axis permutation and shear mapping a ray onto +Z, shared by every triangle test of that ray.

    The transform is a function of the ray alone, so two triangles sharing an edge project that edge's vertices to
    bit-identical coordinates, leaving their tests of it identical up to the order of the two products. Permuting the
    largest absolute direction component last bounds the shear, and swapping the first two axes when that component is
    negative preserves the triangle winding.

    Returns
    -------
    axes : gs.qd_ivec3
        Permuted axis indices: the last one carries the ray, the first two span the plane the edge tests run in.
    shear : qd.math.vec3
        Shear along the first two axes, then the 1 / ray_dir[axes[2]] scale keeping hit distances in ray_dir units.
    is_valid : bool
        False for a direction too short to define a ray, whose traversal must report no hit.
    """
    dir_abs = qd.abs(ray_dir)
    kz = 0
    if dir_abs[1] > dir_abs[0]:
        kz = 1
    if dir_abs[2] > dir_abs[kz]:
        kz = 2
    kx = (kz + 1) % 3
    ky = (kx + 1) % 3
    if ray_dir[kz] < 0.0:
        k_swap = kx
        kx = ky
        ky = k_swap

    shear = qd.math.vec3(0.0, 0.0, 0.0)
    is_valid = dir_abs[kz] > eps
    if is_valid:
        shear = qd.math.vec3(ray_dir[kx], ray_dir[ky], 1.0) / ray_dir[kz]

    return gs.qd_ivec3(kx, ky, kz), shear, is_valid


@qd.func
def ray_triangle_intersection(
    axes: qd.types.vector(3),
    ray_start: qd.types.vector(3),
    shear: qd.types.vector(3),
    v0: qd.types.vector(3),
    v1: qd.types.vector(3),
    v2: qd.types.vector(3),
    eps: float,
):
    """
    Watertight ray-triangle intersection in the ray frame produced by ray_projection.

    The ray is inside the triangle when its three edge tests agree in sign, a test within its own rounding bound
    counting for either side. A ray crossing an edge shared by two triangles is therefore taken by both of them rather
    than dropped through the surface, which is what an inside test on barycentric coordinates does whenever rounding
    places the ray just outside both neighbours at once. The triangles are widened by the uncertainty of their own edge
    tests, and by no more than that.

    Returns
    -------
    hit_distance : gs.qd_float
        Distance along the ray to the intersection, -1.0 if the ray misses the triangle.
    """
    hit_distance = gs.qd_float(-1.0)

    # Vertices relative to the ray origin, sheared so the ray becomes +Z and the edge tests become 2D. Each of the
    # three vectors below holds one projected coordinate of all three vertices.
    verts_rel = qd.Matrix.cols([v0 - ray_start, v1 - ray_start, v2 - ray_start])
    verts_along = verts_rel[axes[2], :]
    verts_x = verts_rel[axes[0], :] - shear[0] * verts_along
    verts_y = verts_rel[axes[1], :] - shear[1] * verts_along
    verts_z = shear[2] * verts_along

    # Twice the signed area the ray forms with each edge, weighting the vertex that edge faces and vanishing when the
    # ray crosses it, together with the bound its two products may cancel down to. That cancellation reaches the point
    # where the sign comes from rounding alone: each of the two triangles holding an edge rounds its own copy of the
    # value, all the more freely as the backend compiler may contract either product into a fused multiply-add. Hence
    # the bound, which lets an area within it count as a crossing - the verdict both triangles then reach.
    edge_areas = verts_y.cross(verts_x)
    edge_bound = 2.0 * eps * verts_x.norm() * verts_y.norm()

    is_crossing = not ((edge_areas < -edge_bound).any() and (edge_areas > edge_bound).any())
    det = edge_areas.sum()
    if is_crossing and det != 0.0:
        # A zero determinant means the ray runs within the triangle plane, where it has no single crossing point.
        t = edge_areas.dot(verts_z) / det
        if t > eps:
            hit_distance = t

    return hit_distance


@qd.func
def ray_aabb_intersection(
    ray_start: qd.types.vector(3),
    ray_dir: qd.types.vector(3),
    aabb_min: qd.types.vector(3),
    aabb_max: qd.types.vector(3),
    eps: float,
):
    """
    Fast ray-AABB intersection test.

    Culling stays conservative, i.e. a box the ray does touch is never rejected, so that the watertight triangle test
    of ray_triangle_intersection is reached for every crossing of the surface.

    Returns the t value of intersection, or -1.0 if no intersection.
    """
    result = -1.0

    # Use the slab method for ray-AABB intersection. The direction is floored only to keep its reciprocal finite, at a
    # magnitude far below any meaningful direction component: flooring at a comparable magnitude instead would
    # overstate how fast the ray leaves the slab of a near-parallel axis, cutting the interval short of a real hit.
    sign = qd.select(ray_dir >= 0.0, 1.0, -1.0)
    inv_dir = sign / qd.max(qd.abs(ray_dir), eps * eps)

    t1 = (aabb_min - ray_start) * inv_dir
    t2 = (aabb_max - ray_start) * inv_dir

    tmin = qd.min(t1, t2)
    tmax = qd.max(t1, t2)

    t_near = qd.max(tmin.x, tmin.y, tmin.z, gs.qd_float(0.0))
    t_far = qd.min(tmax.x, tmax.y, tmax.z)

    # A masked-out face leaves an inverted AABB (min=+inf, max=-inf) as an "unhittable" sentinel. The slab test alone
    # treats that as covering all space (t_near=0 <= t_far=+inf), so the box must be checked non-empty for the sentinel
    # to be a definitive miss regardless of platform NaN/inf comparison behavior.
    is_non_empty = aabb_min.x <= aabb_max.x and aabb_min.y <= aabb_max.y and aabb_min.z <= aabb_max.z
    # The slab bounds carry the rounding of two operations each, so widening the interval by that much keeps culling
    # conservative: a triangle grazed at a corner of its own box stays a candidate, which is what makes the watertight
    # triangle test of ray_triangle_intersection reach every crossing of the surface.
    if is_non_empty and t_near * (1.0 - 2.0 * eps) <= t_far * (1.0 + 2.0 * eps):
        result = t_near

    return result


@qd.func
def closest_point_on_triangle(
    point: qd.types.vector(3), v0: qd.types.vector(3), v1: qd.types.vector(3), v2: qd.types.vector(3)
) -> qd.types.vector(3):
    """
    Closest point on a triangle to a query point.

    Reference: Christer Ericson, Real-Time Collision Detection section 5.1.5.
    """
    ab = v1 - v0
    ac = v2 - v0
    ap = point - v0

    d1 = ab.dot(ap)
    d2 = ac.dot(ap)

    closest = v0
    if not (d1 <= 0.0 and d2 <= 0.0):
        bp = point - v1
        d3 = ab.dot(bp)
        d4 = ac.dot(bp)

        if d3 >= 0.0 and d4 <= d3:
            closest = v1
        else:
            cp = point - v2
            d5 = ab.dot(cp)
            d6 = ac.dot(cp)

            if d6 >= 0.0 and d5 <= d6:
                closest = v2
            else:
                vc = d1 * d4 - d3 * d2
                if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
                    w = d1 / (d1 - d3)
                    closest = v0 + w * ab
                else:
                    vb = d5 * d2 - d1 * d6
                    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
                        w = d2 / (d2 - d6)
                        closest = v0 + w * ac
                    else:
                        va = d3 * d6 - d5 * d4
                        if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
                            w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
                            closest = v1 + w * (v2 - v1)
                        else:
                            denom = 1.0 / (va + vb + vc)
                            v = vb * denom
                            w = vc * denom
                            closest = v0 + v * ab + w * ac
    return closest


@qd.func
def func_sphere_intersects_aabb(
    center: qd.types.vector(3), radius_sq: float, aabb_min: qd.types.vector(3), aabb_max: qd.types.vector(3)
) -> bool:
    """Return whether the point of a box closest to the center of a sphere lies within the sphere.

    A closest-point search passes its best squared distance as `radius_sq` to cull the boxes beyond it.
    """
    offset = qd.max(aabb_min - center, 0.0) + qd.max(center - aabb_max, 0.0)
    return offset.norm_sqr() <= radius_sq


@qd.func
def triangle_face_normal(v0: qd.types.vector(3), v1: qd.types.vector(3), v2: qd.types.vector(3), eps: float):
    """
    Return the outward unit normal of the triangle (v0, v1, v2) under right-hand winding.

    The norm of the normal is floored at `eps`, which keeps a degenerate triangle finite (see 'qd_normalize').
    """
    return gu.qd_normalize((v1 - v0).cross(v2 - v0), eps)


# Fastcache stays off for a kernel taking a visitor or a target (see RaycastTarget)
@qd.kernel
def kernel_cast_ray(
    envs_idx: qd.types.ndarray(ndim=1),
    max_range: float,
    rays_start: qd.types.ndarray(ndim=2),
    ray_direction: qd.types.ndarray(ndim=1),
    target: qd.template(),
):
    """Cast one ray of common direction against the target of each environment in parallel (see RaycastTarget.cast).

    rays_start[i_b] is the start of the ray in the coordinates of environment i_b, and the closest hit of that
    environment is written to the result of the target (see RaycastResult), the environments outside envs_idx being
    left without hit (geom_idx -1, distance +inf). The viewer alone aggregates across environments.
    """
    ray_dir = qd.Vector([ray_direction[0], ray_direction[1], ray_direction[2]], dt=gs.qd_float)
    for i_b in range(target.hits.geom_idx.shape[0]):
        target.hits.distance[i_b] = qd.math.inf
        target.hits.geom_idx[i_b] = -1
        target.hits.hit_point[i_b] = qd.Vector.zero(gs.qd_float, 3)
        target.hits.normal[i_b] = qd.Vector.zero(gs.qd_float, 3)
    for i_b_ in range(envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        ray_start = qd.Vector([rays_start[i_b, 0], rays_start[i_b, 1], rays_start[i_b, 2]], dt=gs.qd_float)
        # The target records the geom and the normal of the closest hit (see RaycastTarget)
        distance = target.cast(i_b, i_b, ray_start, ray_dir, max_range)
        if target.hits.geom_idx[i_b] >= 0:
            target.hits.distance[i_b] = distance
            target.hits.hit_point[i_b] = ray_start + distance * ray_dir


# Fastcache stays off for a kernel taking a visitor or a target (see RaycastTarget)
@qd.kernel
def kernel_cast_rays(
    points_to_sensor_idx: qd.types.ndarray(ndim=1),
    links_pos: qd.types.ndarray(ndim=3),
    links_quat: qd.types.ndarray(ndim=3),
    ray_starts: qd.types.ndarray(ndim=2),
    ray_directions: qd.types.ndarray(ndim=2),
    max_ranges: qd.types.ndarray(ndim=1),
    no_hit_values: qd.types.ndarray(ndim=1),
    is_world_frame: qd.types.ndarray(ndim=1),
    sensor_cache_offsets: qd.types.ndarray(ndim=1),
    sensor_point_offsets: qd.types.ndarray(ndim=1),
    sensor_point_counts: qd.types.ndarray(ndim=1),
    sensor_return_points: qd.types.ndarray(ndim=1),
    output_hits: qd.types.ndarray(ndim=2),
    target: qd.template(),
    eps: float,
    is_merge: qd.template(),
    is_last: qd.template(),
):
    """Cast every ray of every sensor in every environment against a target (see RaycastTarget.cast).

    The output holds per environment, for each sensor at its cache offset, the hit points of its rays when it returns
    them (three values per ray), then the distances of its rays.

    `is_merge` and `is_last` chain the launches over several targets into one output: the first writes every slot and
    each later one overwrites a slot only with a closer hit. A miss never beats a hit of another launch whatever the
    `no_hit_value` of the sensor (it may lie below the range, e.g. 0 or -1): a miss of a launch before the last seeds
    the slot with the range, which every hit lies strictly below, and the last launch alone stamps `no_hit_value` on
    the slots still holding the range.
    """
    n_envs = output_hits.shape[-1]
    for i_flat in range(ray_starts.shape[0] * n_envs):
        # The environment varies fastest, so that the threads of a warp read the same trees of the meshes
        i_p = i_flat // n_envs
        i_b = i_flat % n_envs
        i_s = points_to_sensor_idx[i_p]
        link_pos = qd.Vector([links_pos[i_b, i_s, 0], links_pos[i_b, i_s, 1], links_pos[i_b, i_s, 2]])
        link_quat = qd.Vector(
            [links_quat[i_b, i_s, 0], links_quat[i_b, i_s, 1], links_quat[i_b, i_s, 2], links_quat[i_b, i_s, 3]]
        )
        ray_start = gu.qd_transform_by_trans_quat(
            qd.Vector([ray_starts[i_p, 0], ray_starts[i_p, 1], ray_starts[i_p, 2]]), link_pos, link_quat
        )
        ray_dir_local = qd.Vector([ray_directions[i_p, 0], ray_directions[i_p, 1], ray_directions[i_p, 2]])
        ray_dir = gu.qd_normalize(gu.qd_transform_by_quat(ray_dir_local, link_quat), eps)
        max_range = max_ranges[i_s]
        distance = target.cast(i_b, i_p, ray_start, ray_dir, max_range)
        is_hit = distance < max_range

        i_p_sensor = i_p - sensor_point_offsets[i_s]
        i_p_offset = sensor_cache_offsets[i_s]
        # The distances follow the block of the points when the sensor returns them
        i_p_dist = i_p_offset + i_p_sensor
        if sensor_return_points[i_s]:
            i_p_dist += sensor_point_counts[i_s] * 3
        if is_hit and (not is_merge or distance < output_hits[i_p_dist, i_b]):
            output_hits[i_p_dist, i_b] = distance
            if sensor_return_points[i_s]:
                hit_point = distance * gu.qd_normalize(ray_dir_local, eps)
                if is_world_frame[i_s]:
                    hit_point = ray_start + distance * ray_dir
                for j in qd.static(range(3)):
                    output_hits[i_p_offset + i_p_sensor * 3 + j, i_b] = hit_point[j]
        elif not is_merge:
            if sensor_return_points[i_s]:
                for j in qd.static(range(3)):
                    output_hits[i_p_offset + i_p_sensor * 3 + j, i_b] = 0.0
            output_hits[i_p_dist, i_b] = no_hit_values[i_s] if is_last else max_range
        elif is_last:
            if output_hits[i_p_dist, i_b] >= max_range:
                output_hits[i_p_dist, i_b] = no_hit_values[i_s]


@qd.kernel(fastcache=True)
def kernel_fit_mesh_instances(
    geoms_link_idx: qd.Tensor,
    links_geom_start: qd.Tensor,
    links_geom_end: qd.Tensor,
    geoms_pos: qd.Tensor,
    geoms_quat: qd.Tensor,
    instances: BVHState,
    meshes: PackedInstances,
    rigid_config: qd.template(),
    bvh_config: qd.template(),
    eps: float,
):
    """Fit the box of each geom of every environment posed in the world from the root of its tree, the leaves of the
    top-level trees, and build them.

    A geom without tree or that the environment does not carry (see func_update_geoms) gets an empty box, which no
    query reaches.
    """
    for i_g, i_b in qd.ndrange(instances.leaves.aabbs_min.shape[1], instances.leaves.aabbs_min.shape[0]):
        aabb_min = qd.Vector([qd.math.inf, qd.math.inf, qd.math.inf], dt=gs.qd_float)
        aabb_max = -aabb_min
        i_tree = meshes.instances_tree_idx[i_g]
        i_l = geoms_link_idx[i_g]
        I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l
        if i_tree >= 0 and links_geom_start[I_l] <= i_g and i_g < links_geom_end[I_l]:
            i_root = meshes.trees_root[i_tree]
            aabb_min, aabb_max = gu.qd_transform_aabb(
                meshes.tree.nodes_min[0, i_root],
                meshes.tree.nodes_max[0, i_root],
                trans=geoms_pos[i_g, i_b],
                quat=geoms_quat[i_g, i_b],
            )
        instances.leaves.aabbs_min[i_b, i_g] = aabb_min
        instances.leaves.aabbs_max[i_b, i_g] = aabb_max
    func_build_bvh(instances.leaves.aabbs_min.shape[0], instances, bvh_config, eps)


@qd.kernel(fastcache=True)
def kernel_fit_world_faces(
    dyn_state: DynState,
    bvh_state: BVHState,
    dyn_info: DynInfo,
    faces: WorldFaces,
    rigid_config: qd.template(),
    bvh_config: qd.template(),
    eps: float,
):
    """Fit the box of each face of every environment in the world, the leaves of the trees of the environments, and
    build them.

    A face of a visual geom that the environment does not carry gets an empty box, which no query reaches.
    """
    for i_f, i_b in qd.ndrange(bvh_state.leaves.aabbs_min.shape[1], bvh_state.leaves.aabbs_min.shape[0]):
        aabb_min = qd.Vector([qd.math.inf, qd.math.inf, qd.math.inf], dt=gs.qd_float)
        aabb_max = -aabb_min
        i_g = faces.faces_geom_idx[i_f]
        i_l = dyn_info.vgeoms.link_idx[i_g]
        I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l
        if dyn_info.links.vgeom_start[I_l] <= i_g and i_g < dyn_info.links.vgeom_end[I_l]:
            for j in qd.static(range(3)):
                i_v = faces.faces_verts_idx[i_f][j]
                aabb_min = qd.min(aabb_min, dyn_state.vverts.pos[i_v, i_b])
                aabb_max = qd.max(aabb_max, dyn_state.vverts.pos[i_v, i_b])
        bvh_state.leaves.aabbs_min[i_b, i_f] = aabb_min
        bvh_state.leaves.aabbs_max[i_b, i_f] = aabb_max
    func_build_bvh(bvh_state.leaves.aabbs_min.shape[0], bvh_state, bvh_config, eps)


@qd.data_oriented
class RayTriangles:
    """The visitor of the triangles of a tree that a ray crosses, its bound being the distance of the closest crossing.

    The query is the ray in the frame of the tree, with the projection a triangle test needs, computed once per frame
    (see Query and ray_projection). It reads the vertices of each face, `faces_verts_idx`, among `verts`, and the poses
    of the geoms (`geoms_state`). Without `faces_geom_idx`, the tree spans a mesh in the frame of its geom, the vertices
    are those of the meshes, and the geom is the instance i_k the walk entered (see InstancesQuery). With it, the
    vertices are those of each environment in the world, each face giving its geom (see WorldFaces). With `hits`, it
    records the geom and the outward normal in the world of the closest crossing of each environment, the query index
    being the environment (see RaycastResult).
    """

    is_instance_level = False

    def __init__(
        self,
        verts: qd.Tensor,
        faces_verts_idx: qd.Tensor,
        geoms_state,
        eps: qd.Tensor,
        faces_geom_idx: qd.Tensor | None,
        hits: RaycastResult | None,
    ) -> None:
        self.verts = verts
        self.faces_verts_idx = faces_verts_idx
        self.geoms_state = geoms_state
        self.eps = eps
        self.faces_geom_idx = faces_geom_idx
        self.hits = hits
        self.is_world_verts = faces_geom_idx is not None
        self.is_recording_hits = hits is not None
        # The start and the direction of the ray, then its projection (see ray_projection)
        self.Query = qd.types.struct(
            start=qd.types.vector(3, gs.qd_float),
            dir=qd.types.vector(3, gs.qd_float),
            shear=qd.types.vector(3, gs.qd_float),
            axes=qd.types.vector(3, gs.qd_int),
        )

    @qd.func
    def is_node_hit(
        self, i_t: int, i_q: int, node_min: qd.types.vector(3), node_max: qd.types.vector(3), query, bound: float
    ) -> bool:
        aabb_t = ray_aabb_intersection(query.start, query.dir, node_min, node_max, self.eps[None])
        return aabb_t >= 0.0 and aabb_t < bound

    @qd.func
    def is_instance_kept(self, i_t: int, i_q: int, i_k: int) -> bool:
        return True

    @qd.func
    def to_frame(self, query, pos: qd.types.vector(3), quat: qd.types.vector(4)):
        ray_dir = gu.qd_inv_transform_by_quat(query.dir, quat)
        axes, shear, _is_valid = ray_projection(ray_dir, self.eps[None])
        return self.Query(
            start=gu.qd_inv_transform_by_trans_quat(query.start, pos, quat), dir=ray_dir, shear=shear, axes=axes
        )

    @qd.func
    def visit_leaf(self, i_t: int, i_q: int, i_k: int, i_a: int, query, bound: float) -> float:
        tri_verts = qd.Matrix.zero(gs.qd_float, 3, 3)
        for j in qd.static(range(3)):
            i_v = self.faces_verts_idx[i_a][j]
            if qd.static(self.is_world_verts):
                tri_verts[:, j] = self.verts[i_v, i_t]
            else:
                tri_verts[:, j] = self.verts[i_v]
        v0, v1, v2 = tri_verts[:, 0], tri_verts[:, 1], tri_verts[:, 2]
        distance = ray_triangle_intersection(query.axes, query.start, query.shear, v0, v1, v2, self.eps[None])
        if distance >= 0.0 and distance < bound:
            bound = distance
            if qd.static(self.is_recording_hits):
                normal = triangle_face_normal(v0, v1, v2, self.eps[None])
                if qd.static(self.is_world_verts):
                    self.hits.geom_idx[i_q] = self.faces_geom_idx[i_a]
                else:
                    self.hits.geom_idx[i_q] = i_k
                    normal = gu.qd_transform_by_quat(normal, self.geoms_state.quat[i_k, i_t])
                self.hits.normal[i_q] = normal
        return bound


@qd.data_oriented
class ClosestTriangles:
    """The visitor of the triangles of the trees of the geoms within a radius of a point, its bound being the squared
    distance of the closest one.

    The query is the point, in the frame of the geom the walk entered (see InstancesQuery). It reads the vertices of
    the meshes in the frame of their geom, the vertices of each face, the poses of the geoms (`geoms_state`), the
    sensor of each query and the geoms each sensor may reach (see ClosestQueries and CandidateGeoms), and records the
    closest triangle of each query (see ClosestHits).
    """

    is_instance_level = False

    def __init__(
        self,
        verts_init_pos: qd.Tensor,
        faces_verts_idx: qd.Tensor,
        geoms_state,
        eps: qd.Tensor,
        queries: ClosestQueries,
        candidate_geoms: CandidateGeoms,
        hits: ClosestHits,
    ) -> None:
        self.verts_init_pos = verts_init_pos
        self.faces_verts_idx = faces_verts_idx
        self.geoms_state = geoms_state
        self.eps = eps
        self.queries = queries
        self.candidate_geoms = candidate_geoms
        self.hits = hits
        self.Query = qd.types.struct(point=qd.types.vector(3, gs.qd_float))

    @qd.func
    def is_node_hit(
        self, i_t: int, i_q: int, node_min: qd.types.vector(3), node_max: qd.types.vector(3), query, bound: float
    ) -> bool:
        return func_sphere_intersects_aabb(query.point, bound, node_min, node_max)

    @qd.func
    def is_instance_kept(self, i_t: int, i_q: int, i_k: int) -> bool:
        i_s = self.queries.sensor_idx[i_q]
        i_b = i_t if self.candidate_geoms.is_candidate.shape[2] > 1 else 0
        return self.candidate_geoms.is_candidate[i_s, i_k, i_b]

    @qd.func
    def to_frame(self, query, pos: qd.types.vector(3), quat: qd.types.vector(4)):
        return self.Query(point=gu.qd_inv_transform_by_trans_quat(query.point, pos, quat))

    @qd.func
    def visit_leaf(self, i_t: int, i_q: int, i_k: int, i_a: int, query, bound: float) -> float:
        i_vs = self.faces_verts_idx[i_a]
        v0 = self.verts_init_pos[i_vs[0]]
        v1 = self.verts_init_pos[i_vs[1]]
        v2 = self.verts_init_pos[i_vs[2]]
        diff = query.point - closest_point_on_triangle(query.point, v0, v1, v2)
        dist_sq = diff.norm_sqr()
        if dist_sq < bound:
            bound = dist_sq
            normal = triangle_face_normal(v0, v1, v2, self.eps[None])
            self.hits.signed_dist[i_q, i_t] = qd.select(diff.dot(normal) >= 0.0, 1.0, -1.0) * qd.sqrt(dist_sq)
            self.hits.geom_idx[i_q, i_t] = i_k
            self.hits.normal[i_q, i_t] = gu.qd_transform_by_quat(normal, self.geoms_state.quat[i_k, i_t])
        return bound


@qd.data_oriented
class ClosestQuery:
    """The closest-point query of the triangles of the geoms of a target that the candidate geoms of each sensor
    select, a query per point and environment (see ClosestTriangles)."""

    def __init__(self, instances_query: InstancesQuery, instances: BVHState) -> None:
        self.instances_query = instances_query
        self.instances = instances

    @qd.func
    def closest(self, i_t: int, i_q: int, point: qd.types.vector(3), radius: float):
        """Return the triangle closest to the point of query i_q in environment i_t within a radius.

        The result is the signed distance (negative when the point lies behind the face of the triangle), the geom and
        the outward normal in the world, `radius`, -1 and zero without triangle within the radius.
        """
        self.instances_query.inner.hits.signed_dist[i_q, i_t] = radius
        self.instances_query.inner.hits.geom_idx[i_q, i_t] = -1
        self.instances_query.inner.hits.normal[i_q, i_t] = qd.Vector.zero(gs.qd_float, 3)
        func_bvh_traverse(
            i_t,
            i_q,
            i_root=0,
            leaf_offset=self.instances.tree.leaves_idx.shape[1] - 1,
            leaf_base=0,
            query=self.instances_query.Query(point=point),
            bound=radius * radius,
            tree=self.instances.tree,
            visitor=self.instances_query,
        )
        return (
            self.instances_query.inner.hits.signed_dist[i_q, i_t],
            self.instances_query.inner.hits.geom_idx[i_q, i_t],
            self.instances_query.inner.hits.normal[i_q, i_t],
        )


# FIXME: quadrants#1039 - the fastcache key misses the fields a data_oriented visitor or target reads through its
# methods, so the kernels taking one go without fastcache, their Python front end compiled again in every process.
@qd.data_oriented
class RaycastTarget:
    """The collision or the visual geoms of a solver as a target of ray casts and closest-point queries.

    The trees of their meshes (see get_solver_meshes) lie under a top-level tree per environment over the geoms. With
    `hits`, its ray casts record the geom and the normal of the closest hit of each environment there.
    """

    def __init__(
        self, solver: "KinematicSolver", is_visual: bool, meshes: PackedInstances, hits: RaycastResult | None
    ) -> None:
        self.solver = solver
        self.is_visual = is_visual
        self.meshes = meshes
        self.hits = hits
        self.eps = solver.rigid_info.EPS
        dyn_info, dyn_state = solver.dyn_info, solver.dyn_state
        self.geoms_info = dyn_info.vgeoms if is_visual else dyn_info.geoms
        self.geoms_state = dyn_state.vgeoms if is_visual else dyn_state.geoms
        # The arrays alone, the collision and the visual structs naming their fields differently
        self.verts_init_pos = dyn_info.vverts.init_pos if is_visual else dyn_info.verts.init_pos
        self.faces_verts_idx = dyn_info.vfaces.vverts_idx if is_visual else dyn_info.faces.verts_idx
        bvh = get_bvh_data(solver._B, meshes.instances_tree_idx.shape[0])
        self.tree_state = bvh.state
        self.tree_config = bvh.config
        self.ray_query = InstancesQuery(
            RayTriangles(
                self.verts_init_pos, self.faces_verts_idx, self.geoms_state, self.eps, faces_geom_idx=None, hits=hits
            ),
            meshes,
            self.geoms_state,
        )

    def update(self) -> None:
        """Fit the top-level trees to the current poses of the geoms."""
        if self.is_visual:
            self.solver.update_forward_pos()
            self.solver.update_vgeoms()
            dyn_info = self.solver.dyn_info
            kernel_fit_mesh_instances(
                self.geoms_info.link_idx,
                dyn_info.links.vgeom_start,
                dyn_info.links.vgeom_end,
                self.geoms_state.pos,
                self.geoms_state.quat,
                self.tree_state,
                self.meshes,
                self.solver.rigid_config,
                self.tree_config,
                eps=gs.EPS,
            )
        else:
            # The forward kinematics of the rigid solver refits the box of each geom with its pose (see
            # func_update_geom_aabb), before the sensors and the viewer read it.
            kernel_build_instances_tree(
                i_k_excluded=-1,
                instances_aabb_min=self.geoms_state.aabb_min,
                instances_aabb_max=self.geoms_state.aabb_max,
                bvh_state=self.tree_state,
                packed_instances=self.meshes,
                bvh_config=self.tree_config,
                eps=gs.EPS,
            )

    def build_closest_query(self, queries_sensor_idx: np.ndarray, candidate_geoms: CandidateGeoms) -> ClosestQuery:
        """Return the closest-point query of the triangles of the geoms, of the given sensor per query, among the
        candidate geoms of each sensor."""
        queries = ClosestQueries(sizes=dict(closest_queries=len(queries_sensor_idx)))
        queries.sensor_idx.from_numpy(queries_sensor_idx)
        hits = ClosestHits(sizes=dict(closest_queries=len(queries_sensor_idx), envs=self.solver._B))
        closest_triangles = ClosestTriangles(
            self.verts_init_pos, self.faces_verts_idx, self.geoms_state, self.eps, queries, candidate_geoms, hits
        )
        return ClosestQuery(InstancesQuery(closest_triangles, self.meshes, self.geoms_state), self.tree_state)

    @qd.func
    def cast(
        self, i_t: int, i_q: int, ray_start: qd.types.vector(3), ray_dir: qd.types.vector(3), max_range: float
    ) -> float:
        """Cast ray i_q against the geoms of environment i_t, and return the distance of its closest hit, `max_range`
        without hit or for a direction too short to define a ray."""
        distance = max_range
        axes, shear, is_valid = ray_projection(ray_dir, self.eps[None])
        if is_valid:
            distance = func_bvh_traverse(
                i_t,
                i_q,
                i_root=0,
                leaf_offset=self.tree_state.tree.leaves_idx.shape[1] - 1,
                leaf_base=0,
                query=self.ray_query.Query(start=ray_start, dir=ray_dir, shear=shear, axes=axes),
                bound=max_range,
                tree=self.tree_state.tree,
                visitor=self.ray_query,
            )
        return distance


@qd.data_oriented
class RaycastWorldTarget(RaycastTarget):
    """The visual geoms of a solver opted into raycasting whose vertices are set per environment in the world (see
    set_vverts) as a target of ray casts: a tree per environment over their faces.

    The trees are rebuilt when a change of the geometry of the solver is pending and at the first update. With `hits`,
    its ray casts record the geom and the normal of the closest hit of each environment there.
    """

    def __init__(self, solver: "KinematicSolver", faces_idx: np.ndarray, hits: RaycastResult | None) -> None:
        self.solver = solver
        self.is_visual = True
        self.hits = hits
        self.eps = solver.rigid_info.EPS
        dyn_info = solver.dyn_info
        vverts_state_idx = qd_to_numpy(dyn_info.vverts.vverts_state_idx, transpose=True)
        faces_vverts_idx = qd_to_numpy(dyn_info.vfaces.vverts_idx, transpose=True)
        self.faces = WorldFaces(sizes=dict(world_faces=len(faces_idx)))
        self.faces.faces_verts_idx.from_numpy(vverts_state_idx[faces_vverts_idx[faces_idx]])
        self.faces.faces_geom_idx.from_numpy(qd_to_numpy(dyn_info.vfaces.vgeom_idx, transpose=True)[faces_idx])
        bvh = get_bvh_data(solver._B, len(faces_idx))
        self.tree_state = bvh.state
        self.tree_config = bvh.config
        self.ray_query = RayTriangles(
            solver.dyn_state.vverts.pos,
            self.faces.faces_verts_idx,
            solver.dyn_state.vgeoms,
            self.eps,
            faces_geom_idx=self.faces.faces_geom_idx,
            hits=hits,
        )
        self.subscriber = Subscriber(to=frozenset({StateChange.GEOMETRY}))
        solver.subscribe(self.subscriber)
        self.is_stale = True

    def update(self) -> None:
        """Rebuild the trees if the vertices may have changed since the last build."""
        if not (self.is_stale or self.subscriber.pending):
            return
        self.subscriber.clear()
        self.is_stale = False
        kernel_fit_world_faces(
            self.solver.dyn_state,
            self.tree_state,
            self.solver.dyn_info,
            self.faces,
            self.solver.rigid_config,
            self.tree_config,
            eps=gs.EPS,
        )


def get_raycast_targets(
    solver: "KinematicSolver", is_visual: bool, hits: RaycastResult | None = None
) -> list[RaycastTarget]:
    """Return the targets of the collision geoms of a solver, or of its visual geoms opted into raycasting, recording
    the geom and the normal of the closest hit of each environment in `hits` if given.

    The visual geoms whose vertices are set per environment (see set_vverts) form a target of their own. A target
    without face is left out.
    """
    dyn_info = solver.dyn_info
    if not is_visual:
        if dyn_info.faces.verts_idx.shape[0] == 0:
            return []
        return [RaycastTarget(solver, is_visual, get_solver_meshes(solver, is_visual), hits)]
    faces_vverts_idx = qd_to_numpy(dyn_info.vfaces.vverts_idx, transpose=True)
    faces_is_world = qd_to_numpy(dyn_info.vverts.vverts_state_idx, transpose=True)[faces_vverts_idx[:, 0]] >= 0
    faces_is_used = tensor_to_array(solver.vfaces_raycast_mask, dtype=bool)
    faces_is_mesh = faces_is_used & ~faces_is_world
    targets = []
    if faces_is_mesh.any():
        targets.append(RaycastTarget(solver, is_visual, get_solver_meshes(solver, is_visual, faces_is_mesh), hits))
    if (faces_is_used & faces_is_world).any():
        targets.append(RaycastWorldTarget(solver, np.flatnonzero(faces_is_used & faces_is_world), hits))
    return targets
