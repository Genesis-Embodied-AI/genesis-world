import numpy as np

import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
from genesis.engine.bvh import (
    BVHState,
    InstancesQuery,
    PackedInstances,
    func_build_bvh,
    func_bvh_traverse,
    get_bvh_data,
    get_meshes_trees,
    kernel_build_instances_tree,
)
from genesis.engine.render_context import (
    DynamicMesh,
    MeshGeometry,
    ScopedRenderContext,
    SegmentationKeys,
    StaticMeshBatch,
)
from genesis.engine.state_change import StateChange
from genesis.utils.data import ENVS, Data, DataKind, Size, field
from genesis.utils.misc import tensor_to_array


class RaycastInfo(Data, kind=DataKind.INFO):
    """The constants a ray cast against a target reads.

    The target holds them in a struct of its own, so that the kernels casting rays take a single scalar array rather
    than the struct of another component.
    """

    EPS: qd.Tensor = field(float, shape=())


class RaycastResult(Data, kind=DataKind.SCRATCH, shape=(ENVS,)):
    """
    The nearest hit of a ray cast against the rigid geoms in each environment: its distance, geom, point and normal.
    """

    distance: qd.Tensor = field(float)
    geom_idx: qd.Tensor = field(int)
    hit_point: qd.Tensor = field((float, 3))
    normal: qd.Tensor = field((float, 3))


# Faces of the dynamic meshes posed in the world, and the meshes
WORLD_FACES = Size("world_faces")
WORLD_MESHES = Size("world_meshes")


class WorldFaces(Data, kind=DataKind.CONSTANT, shape=(WORLD_FACES,)):
    """The faces of the meshes whose vertices are placed per environment in the world (see DynamicMesh)."""

    # Vertices of each face among those of the array holding the meshes, and its mesh
    faces_verts_idx: qd.Tensor = field((int, 3))
    faces_mesh_idx: qd.Tensor = field(int)


class WorldMeshes(Data, kind=DataKind.CONSTANT, shape=(WORLD_MESHES, ENVS)):
    """Whether each environment holds each mesh whose vertices are placed in the world (see DynamicMesh)."""

    is_active: qd.Tensor = field(bool)


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
    instances_is_active: qd.Tensor,
    instances_pos: qd.Tensor,
    instances_quat: qd.Tensor,
    instances: BVHState,
    meshes: PackedInstances,
    bvh_config: qd.template(),
    eps: float,
    has_is_active: qd.template(),
):
    """Fit the box of each instance of every environment posed in the world from the root of its tree, the leaves of
    the top-level trees, and build them.

    An instance without tree, or that the environment does not hold as `instances_is_active` tells with
    `has_is_active`, gets an empty box, which no query reaches.
    """
    for i_k, i_b in qd.ndrange(instances.leaves.aabbs_min.shape[1], instances.leaves.aabbs_min.shape[0]):
        aabb_min = qd.Vector([qd.math.inf, qd.math.inf, qd.math.inf], dt=gs.qd_float)
        aabb_max = -aabb_min
        i_tree = meshes.instances_tree_idx[i_k]
        is_placed = i_tree >= 0
        if qd.static(has_is_active):
            is_placed = is_placed and instances_is_active[i_k, i_b]
        if is_placed:
            i_root = meshes.trees_root[i_tree]
            aabb_min, aabb_max = gu.qd_transform_aabb(
                meshes.tree.nodes_min[0, i_root],
                meshes.tree.nodes_max[0, i_root],
                trans=instances_pos[i_k, i_b],
                quat=instances_quat[i_k, i_b],
            )
        instances.leaves.aabbs_min[i_b, i_k] = aabb_min
        instances.leaves.aabbs_max[i_b, i_k] = aabb_max
    func_build_bvh(instances.leaves.aabbs_min.shape[0], instances, bvh_config, eps)


@qd.kernel(fastcache=True)
def kernel_fit_world_faces(
    verts_pos: qd.Tensor,
    bvh_state: BVHState,
    faces: WorldFaces,
    meshes: WorldMeshes,
    bvh_config: qd.template(),
    eps: float,
):
    """Fit the box of each face of every environment in the world, the leaves of the trees of the environments, and
    build them.

    A face of a mesh that the environment does not hold gets an empty box, which no query reaches.
    """
    for i_f, i_b in qd.ndrange(bvh_state.leaves.aabbs_min.shape[1], bvh_state.leaves.aabbs_min.shape[0]):
        aabb_min = qd.Vector([qd.math.inf, qd.math.inf, qd.math.inf], dt=gs.qd_float)
        aabb_max = -aabb_min
        i_m = faces.faces_mesh_idx[i_f]
        if meshes.is_active[i_m, i_b]:
            for j in qd.static(range(3)):
                i_v = faces.faces_verts_idx[i_f][j]
                aabb_min = qd.min(aabb_min, verts_pos[i_v, i_b])
                aabb_max = qd.max(aabb_max, verts_pos[i_v, i_b])
        bvh_state.leaves.aabbs_min[i_b, i_f] = aabb_min
        bvh_state.leaves.aabbs_max[i_b, i_f] = aabb_max
    func_build_bvh(bvh_state.leaves.aabbs_min.shape[0], bvh_state, bvh_config, eps)


@qd.data_oriented
class RayTriangles:
    """The visitor of the triangles of a tree that a ray crosses, its bound being the distance of the closest crossing.

    The query is the ray in the frame of the tree, with the projection a triangle test needs, computed once per frame
    (see Query and ray_projection). It reads the vertices of each face, `faces_verts_idx`, among `verts`, and the poses
    of the meshes (`poses`). Without `faces_mesh_idx`, the tree spans a mesh in the frame of its pose, the
    vertices are those of the meshes, and the mesh is the instance i_k the walk entered (see InstancesQuery). With it,
    the vertices are those of each environment in the world, each face giving its mesh (see WorldFaces). With `hits`, it
    records the mesh and the outward normal in the world of the closest crossing of each environment, the query index
    being the environment (see RaycastResult).
    """

    is_instance_level = False

    def __init__(
        self,
        verts: qd.Tensor,
        faces_verts_idx: qd.Tensor,
        poses: "MeshesPoses | None",
        eps: qd.Tensor,
        faces_mesh_idx: qd.Tensor | None,
        hits: RaycastResult | None,
    ) -> None:
        self.verts = verts
        self.faces_verts_idx = faces_verts_idx
        self.poses = poses
        self.eps = eps
        self.faces_mesh_idx = faces_mesh_idx
        self.hits = hits
        self.is_world_verts = faces_mesh_idx is not None
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
        self,
        i_t: int,
        i_q: int,
        i_n: int,
        node_min: qd.types.vector(3),
        node_max: qd.types.vector(3),
        query,
        bound: float,
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
                    self.hits.geom_idx[i_q] = self.faces_mesh_idx[i_a]
                else:
                    self.hits.geom_idx[i_q] = i_k
                    normal = gu.qd_transform_by_quat(normal, self.poses.quat[i_k, i_t])
                self.hits.normal[i_q] = normal
        return bound


@qd.data_oriented
class ClosestTriangles:
    """The visitor of the triangles of the trees of the geoms within a radius of a point, its bound being the squared
    distance of the closest one.

    The query is the point, in the frame of the geom the walk entered (see InstancesQuery). It reads the vertices of
    the meshes in the frame of their geom, the vertices of each face, the poses of the geoms (`poses`), the
    sensor of each query and the geoms each sensor may reach (see ClosestQueries and CandidateGeoms), and records the
    closest triangle of each query (see ClosestHits).
    """

    is_instance_level = False

    def __init__(
        self,
        verts_init_pos: qd.Tensor,
        faces_verts_idx: qd.Tensor,
        poses: "MeshesPoses",
        eps: qd.Tensor,
        queries: ClosestQueries,
        candidate_geoms: CandidateGeoms,
        hits: ClosestHits,
    ) -> None:
        self.verts_init_pos = verts_init_pos
        self.faces_verts_idx = faces_verts_idx
        self.poses = poses
        self.eps = eps
        self.queries = queries
        self.candidate_geoms = candidate_geoms
        self.hits = hits
        self.Query = qd.types.struct(point=qd.types.vector(3, gs.qd_float))

    @qd.func
    def is_node_hit(
        self,
        i_t: int,
        i_q: int,
        i_n: int,
        node_min: qd.types.vector(3),
        node_max: qd.types.vector(3),
        query,
        bound: float,
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
            self.hits.normal[i_q, i_t] = gu.qd_transform_by_quat(normal, self.poses.quat[i_k, i_t])
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


@qd.data_oriented
class MeshesPoses:
    """The world pose of each mesh of a target in each environment, the arrays its StaticMeshBatch views."""

    def __init__(self, pos: qd.Tensor, quat: qd.Tensor) -> None:
        self.pos = pos
        self.quat = quat


# FIXME: quadrants#1039 - the fastcache key misses the fields a data_oriented visitor or target reads through its
# methods, so the kernels taking one go without fastcache, their Python front end compiled again in every process.
@qd.data_oriented
class RaycastTarget:
    """The meshes of a StaticMeshBatch of the render context as a target of ray casts and closest-point queries.

    The trees of the meshes lie under a top-level tree per environment over them, each mesh an instance. The hits name a
    mesh by its position in the batch, whose segmentation keys are those of `meshes_seg_keys`, None for a mesh the ray
    casting leaves out. `is_visual` tells whether the meshes are visual meshes, which their scoped context refreshes, or
    collision meshes, posed by the forward kinematics of their solver. With `hits`, its ray casts record the mesh and
    the normal of the closest hit of each environment there. 'update' fits the top-level trees to the current poses.
    """

    def __init__(
        self,
        scoped_context: ScopedRenderContext,
        batch: StaticMeshBatch,
        geometries: "tuple[MeshGeometry | None, ...]",
        is_visual: bool,
        hits: RaycastResult | None,
    ) -> None:
        self.scoped_context = scoped_context
        self.is_visual = is_visual
        self.hits = hits
        self.meshes_seg_keys: "tuple[SegmentationKeys | None, ...]" = tuple(
            None if geometry is None else geometry.seg_keys for geometry in geometries
        )
        # The batches of the scoped contexts view their pose arrays from their first element on
        refs = batch.refs
        self.poses = MeshesPoses(refs["pos"].array, refs["quat"].array)
        is_active_ref, aabb_min_ref, aabb_max_ref = (refs.get(name) for name in ("is_active", "aabb_min", "aabb_max"))
        self.has_is_active = is_active_ref is not None
        self.meshes_is_active = refs["pos"].array if is_active_ref is None else is_active_ref.array
        self.meshes_aabb = None if aabb_min_ref is None else (aabb_min_ref.array, aabb_max_ref.array)
        self.info = RaycastInfo(values=dict(EPS=gs.EPS))
        self.eps = self.info.EPS
        geometry = next(geometry for geometry in geometries if geometry is not None)
        self.verts_init_pos = geometry.verts_ref.array
        self.faces_verts_idx = geometry.faces_ref.array
        faces_range = tuple(
            (0, 0) if geometry is None else (geometry.faces_ref.cols.start, geometry.faces_ref.cols.stop)
            for geometry in geometries
        )
        self.meshes = get_meshes_trees(scoped_context.owner, self.verts_init_pos, self.faces_verts_idx, faces_range)
        bvh = get_bvh_data(len(batch.envs), len(geometries))
        self.tree_state = bvh.state
        self.tree_config = bvh.config
        self.ray_query = InstancesQuery(
            RayTriangles(
                self.verts_init_pos, self.faces_verts_idx, self.poses, self.eps, faces_mesh_idx=None, hits=hits
            ),
            self.meshes,
            self.poses,
        )

    def update(self) -> None:
        """Fit the top-level trees to the current poses of the meshes, for visual meshes once their scoped context
        refreshed them since the last fit."""
        if self.is_visual:
            self.scoped_context.get_shapes()
            changelog = self.scoped_context.changelog
            if not changelog.is_dirty_since(self, StateChange.GEOMETRY):
                return
            changelog.mark_clean(self)
        else:
            # The collision meshes follow the poses of the geoms, which forward kinematics refreshes with the links
            solver = self.scoped_context.owner
            solver.changelog.refresh(solver.dyn_state.links.pos)
        if self.meshes_aabb is not None:
            # The boxes of the meshes in the world, which the poses keep current, give the leaves of the trees as is
            kernel_build_instances_tree(
                i_k_excluded=-1,
                instances_aabb_min=self.meshes_aabb[0],
                instances_aabb_max=self.meshes_aabb[1],
                bvh_state=self.tree_state,
                packed_instances=self.meshes,
                bvh_config=self.tree_config,
                eps=gs.EPS,
            )
            return
        kernel_fit_mesh_instances(
            self.meshes_is_active,
            self.poses.pos,
            self.poses.quat,
            self.tree_state,
            self.meshes,
            self.tree_config,
            eps=gs.EPS,
            has_is_active=self.has_is_active,
        )

    def build_closest_query(self, queries_sensor_idx: np.ndarray, candidate_geoms: CandidateGeoms) -> ClosestQuery:
        """Return the closest-point query of the triangles of the meshes, of the given sensor per query, among the
        candidate meshes of each sensor."""
        queries = ClosestQueries(sizes=dict(closest_queries=len(queries_sensor_idx)))
        queries.sensor_idx.from_numpy(queries_sensor_idx)
        hits = ClosestHits(sizes=dict(closest_queries=len(queries_sensor_idx), envs=self.poses.pos.shape[1]))
        closest_triangles = ClosestTriangles(
            self.verts_init_pos, self.faces_verts_idx, self.poses, self.eps, queries, candidate_geoms, hits
        )
        return ClosestQuery(InstancesQuery(closest_triangles, self.meshes, self.poses), self.tree_state)

    @qd.func
    def cast(
        self, i_t: int, i_q: int, ray_start: qd.types.vector(3), ray_dir: qd.types.vector(3), max_range: float
    ) -> float:
        """Cast ray i_q against the meshes of environment i_t, and return the distance of its closest hit, `max_range`
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
    """Ray-cast target over the dynamic meshes of a scoped context, placed per environment in the world.

    It holds a tree per environment over their faces, which the hits name by mesh (see 'RaycastTarget'). The trees are
    built at the first update, then again whenever the scoped context refreshed the meshes.
    """

    def __init__(
        self, scoped_context: ScopedRenderContext, dynamic_meshes: tuple[DynamicMesh, ...], hits: RaycastResult | None
    ):
        self.scoped_context = scoped_context
        self.is_visual = True
        self.hits = hits
        self.meshes_seg_keys = tuple(mesh.seg_keys for mesh in dynamic_meshes)
        self.info = RaycastInfo(values=dict(EPS=gs.EPS))
        self.eps = self.info.EPS
        # The meshes of a scoped context view the same array, as vertices of each environment in the world
        self.verts_pos = dynamic_meshes[0].refs["verts"].array
        n_envs = len(dynamic_meshes[0].envs)
        faces_verts_idx, faces_mesh_idx = [], []
        meshes_is_active = np.ones((len(dynamic_meshes), n_envs), dtype=gs.np_bool)
        for i_m, mesh in enumerate(dynamic_meshes):
            faces_verts_idx.append(tensor_to_array(mesh.faces) + mesh.refs["verts"].cols.start)
            faces_mesh_idx.append(np.full(len(mesh.faces), i_m, dtype=gs.np_int))
            # The environments holding the visual geom of a heterogeneous variant are constant
            if mesh.is_active is not None:
                meshes_is_active[i_m] = tensor_to_array(mesh.is_active, dtype=gs.np_bool)
        faces_verts_idx = np.concatenate(faces_verts_idx, dtype=gs.np_int)
        self.faces = WorldFaces(sizes=dict(world_faces=len(faces_verts_idx)))
        self.faces.faces_verts_idx.from_numpy(faces_verts_idx)
        self.faces.faces_mesh_idx.from_numpy(np.concatenate(faces_mesh_idx))
        self.meshes = WorldMeshes(sizes=dict(world_meshes=len(dynamic_meshes), envs=n_envs))
        self.meshes.is_active.from_numpy(meshes_is_active)
        bvh = get_bvh_data(n_envs, len(faces_verts_idx))
        self.tree_state = bvh.state
        self.tree_config = bvh.config
        self.ray_query = RayTriangles(
            self.verts_pos,
            self.faces.faces_verts_idx,
            None,
            self.eps,
            faces_mesh_idx=self.faces.faces_mesh_idx,
            hits=hits,
        )

    def update(self) -> None:
        """Build the trees of the environments over the faces at their current positions, once the scoped context of the
        meshes refreshed them since the last build."""
        self.scoped_context.get_shapes()
        changelog = self.scoped_context.changelog
        if not changelog.is_dirty_since(self, StateChange.GEOMETRY):
            return
        changelog.mark_clean(self)
        kernel_fit_world_faces(self.verts_pos, self.tree_state, self.faces, self.meshes, self.tree_config, eps=gs.EPS)


def get_raycast_targets(
    scoped_contexts: tuple[ScopedRenderContext, ...],
    is_visual: bool,
    hits: RaycastResult | None = None,
) -> "list[RaycastTarget | RaycastWorldTarget]":
    """Return the targets of the collision geoms or of the visual meshes the given scoped contexts of solvers draw.

    The meshes are read from the render context. A target without mesh is left out.

    Parameters
    ----------
    scoped_contexts : tuple[ScopedRenderContext, ...]
        The scoped contexts of the solvers.
    is_visual : bool
        Whether the targets are the static and dynamic meshes of the scene as it looks of the entities opted into visual
        raycasting (see ``gs.entities.Rigid.use_visual_raycasting``), rather than the collision representation of each,
        which is then that of a rigid solver.
    hits : RaycastResult | None, optional
        Where the targets record the mesh and the normal of the closest hit of each environment. Defaults to None.
    """
    targets: "list[RaycastTarget | RaycastWorldTarget]" = []
    for scoped_context in scoped_contexts:
        shapes = scoped_context.get_shapes(debug_vis_mode=None if is_visual else "collision")
        for batch in shapes:
            if isinstance(batch, StaticMeshBatch):
                geometries = tuple(
                    geometry if geometry.is_ray_target or not is_visual else None for geometry in batch.geometries
                )
                if any(geometry is not None for geometry in geometries):
                    targets.append(RaycastTarget(scoped_context, batch, geometries, is_visual, hits))
        dynamic_meshes = tuple(shape for shape in shapes if isinstance(shape, DynamicMesh) and shape.is_ray_target)
        if dynamic_meshes:
            targets.append(RaycastWorldTarget(scoped_context, dynamic_meshes, hits))
    return targets
