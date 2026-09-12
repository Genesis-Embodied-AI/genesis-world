import quadrants as qd

import genesis as gs
import genesis.utils.array_class as array_class
import genesis.utils.geom as gu
from genesis.engine.bvh import STACK_SIZE
from genesis.engine.solvers.rigid.rigid_solver import func_update_all_verts


# FIXME: get_triangle_vertices/bvh_ray_cast/update_aabbs duplicate their visual counterparts below. The two paths
# differ only in the leaf-data fetch (fixed/free verts split vs vverts_state_idx + FK fallback) and in the dataclass
# shapes of geoms_state vs vgeoms_state. Quadrants does not currently support a qd.func arg that accepts either of
# two dataclasses with different field sets, so we cannot factor the BVH traversal into a single shared kernel; the
# kernel-call argument-fusion step strictly matches the annotated dataclass shape and rejects either union typing or
# qd.template() for dataclass-typed args. Until Quadrants gains generic-dataclass dispatch, the visual variants below
# stay as parallel copies.


@qd.func
def get_triangle_vertices(i_f: int, i_b: int, dyn_state: array_class.DynState, dyn_info: array_class.DynInfo):
    """
    Get the three vertices of a triangle in world space.

    Returns
    -------
    tri_vertices : qd.Matrix
        3x3 matrix where each column is a vertex position.
    """
    tri_vertices = qd.Matrix.zero(gs.qd_float, 3, 3)
    for i in qd.static(range(3)):
        i_v = dyn_info.faces.verts_idx[i_f][i]
        i_fv = dyn_info.verts.verts_state_idx[i_v]
        if dyn_info.verts.is_fixed[i_v]:
            tri_vertices[:, i] = dyn_state.fixed_verts.pos[i_fv]
        else:
            tri_vertices[:, i] = dyn_state.free_verts.pos[i_fv, i_b]
    return tri_vertices


@qd.func
def bvh_ray_cast(
    i_t: int,
    i_b: int,
    i_s: int,  # sensor index selecting the link_excluded row
    ray_start: qd.types.vector(3),
    ray_dir: qd.types.vector(3),
    max_range: float,
    bvh_nodes: qd.template(),
    bvh_morton_codes: qd.template(),
    link_excluded: qd.types.ndarray(ndim=2),  # [n_sensors, n_links] bool - 0 marks an excluded link
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    eps: float,
    is_link_excluded: qd.template(),  # compile-time: whether any sensor excludes links
):
    """
    Cast a ray through a BVH and find the closest intersection.

    ``i_t`` selects the BVH tree slot (nodes / morton codes) and ``i_b`` the env whose verts back the leaf
    triangles: ``i_t == i_b`` for a per-env BVH, while a grouped static BVH shared by several envs routes ``i_t``
    through ``env_bvh_idx`` (the routed envs have bit-identical verts, so each env reads its own).

    ``i_s`` selects the sensor whose ``link_excluded`` row a leaf is checked against; when ``is_link_excluded``
    is False (compile time) that check is compiled out, so non-filtering sensors pay no per-leaf cost.

    Returns
    -------
    hit_face : gs.qd_int
        index of the hit (global) triangle (-1 if no hit)
    hit_distance : gs.qd_float
        distance to hit point (unchanged max_range if no hit)
    hit_normal : qd.math.vec3
        normal vector at hit point (zero vector if no hit)
    """
    # The BVH's own leaf count. For a subset BVH this is the subset size, smaller than the solver's face count.
    n_triangles = bvh_morton_codes.shape[1]

    hit_face = -1
    closest_distance = gs.qd_float(max_range)
    hit_normal = qd.math.vec3(0.0, 0.0, 0.0)
    # Declared before the compile-time branch: stays 0 (not excluded) when
    # is_link_excluded is False, keeping that path a no-op.
    is_excluded = gs.qd_int(0)

    axes, shear, is_valid_dir = ray_projection(ray_dir, eps)

    # Stack for non-recursive BVH traversal
    node_stack = qd.Vector.zero(gs.qd_int, qd.static(STACK_SIZE))
    node_stack[0] = 0  # Start at root node
    stack_idx = 1
    if not is_valid_dir:
        # A direction too short to define a ray leaves the stack empty, reporting no hit.
        stack_idx = 0

    while stack_idx > 0:
        stack_idx -= 1
        node_idx = node_stack[stack_idx]

        node = bvh_nodes[i_t, node_idx]

        # Check if ray hits the node's bounding box
        aabb_t = ray_aabb_intersection(ray_start, ray_dir, node.bound.min, node.bound.max, eps)

        if aabb_t >= 0.0 and aabb_t < closest_distance:
            if node.left == -1:  # Leaf node
                # The leaf payload carries the global face index; see kernel_remap_leaf_faces.
                sorted_leaf_idx = node_idx - (n_triangles - 1)
                i_f = qd.cast(bvh_morton_codes[i_t, sorted_leaf_idx][1], gs.qd_int)

                # Get triangle vertices
                tri_vertices = get_triangle_vertices(i_f, i_b, dyn_state, dyn_info)
                v0, v1, v2 = tri_vertices[:, 0], tri_vertices[:, 1], tri_vertices[:, 2]

                # Perform ray-triangle intersection
                hit_distance = ray_triangle_intersection(axes, ray_start, shear, v0, v1, v2, eps)

                if qd.static(is_link_excluded):
                    # A ray must not hit faces owned by this sensor's excluded
                    # links (e.g. the robot's own links for a terrain scan).
                    # mask[row, link] == 0 marks an excluded link.
                    is_excluded = link_excluded[i_s, dyn_info.geoms.link_idx[dyn_info.faces.geom_idx[i_f]]] == 0

                if is_excluded == 0 and hit_distance >= 0.0 and hit_distance < closest_distance:
                    closest_distance = hit_distance
                    hit_face = i_f
                    hit_normal = triangle_face_normal(v0, v1, v2)
            else:  # Internal node
                # Push children onto stack
                if stack_idx < qd.static(STACK_SIZE - 2):
                    node_stack[stack_idx] = node.left
                    node_stack[stack_idx + 1] = node.right
                    stack_idx += 2

    return hit_face, closest_distance, hit_normal


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
def triangle_face_normal(v0: qd.types.vector(3), v1: qd.types.vector(3), v2: qd.types.vector(3)) -> qd.types.vector(3):
    """Outward unit normal of the triangle (v0, v1, v2) under right-hand winding."""
    return (v1 - v0).cross(v2 - v0).normalized()


@qd.func
def update_face_aabb(
    i_t: int,
    i_a: int,
    i_f: int,
    i_b: int,
    dyn_state: array_class.DynState,
    aabb_state: qd.template(),
    dyn_info: array_class.DynInfo,
    rigid_config: qd.template(),
):
    """Fit AABB slot i_a of tree slot i_t to face i_f from env i_b's current vertex positions.

    ``i_t == i_b`` for a per-env BVH; a grouped static BVH builds each tree slot from its representative env (see
    RaycastContext update).

    A face contributes to env i_b only if its geom lies in that env's active geom range (links_info.geom_start /
    geom_end); otherwise its AABB is left inverted (unhittable) and skipped by ray queries. For a homogeneous solver
    every geom is always in range, so this never excludes anything. For a heterogeneous solver, where all envs share
    one vertex buffer but activate different per-env geom ranges, it makes each env cast against only its own variant
    instead of the union of every variant.
    """
    aabb_state.aabbs[i_t, i_a].min.fill(qd.math.inf)
    aabb_state.aabbs[i_t, i_a].max.fill(-qd.math.inf)

    i_g = dyn_info.faces.geom_idx[i_f]
    i_l = dyn_info.geoms.link_idx[i_g]
    I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l
    if dyn_info.links.geom_start[I_l] <= i_g and i_g < dyn_info.links.geom_end[I_l]:
        for i in qd.static(range(3)):
            i_v = dyn_info.faces.verts_idx[i_f][i]
            i_fv = dyn_info.verts.verts_state_idx[i_v]
            if dyn_info.verts.is_fixed[i_v]:
                pos_v = dyn_state.fixed_verts.pos[i_fv]
                aabb_state.aabbs[i_t, i_a].min = qd.min(aabb_state.aabbs[i_t, i_a].min, pos_v)
                aabb_state.aabbs[i_t, i_a].max = qd.max(aabb_state.aabbs[i_t, i_a].max, pos_v)
            else:
                pos_v = dyn_state.free_verts.pos[i_fv, i_b]
                aabb_state.aabbs[i_t, i_a].min = qd.min(aabb_state.aabbs[i_t, i_a].min, pos_v)
                aabb_state.aabbs[i_t, i_a].max = qd.max(aabb_state.aabbs[i_t, i_a].max, pos_v)


@qd.func
def update_aabbs(
    dyn_state: array_class.DynState,
    aabb_state: qd.template(),
    dyn_info: array_class.DynInfo,
    rigid_config: qd.template(),
):
    """Update the per-face collision AABBs of a per-env BVH covering every face in order. See update_face_aabb."""
    for i_b, i_f in qd.ndrange(dyn_state.free_verts.pos.shape[1], dyn_info.faces.verts_idx.shape[0]):
        update_face_aabb(i_b, i_f, i_f, i_b, dyn_state, aabb_state, dyn_info, rigid_config)


@qd.func
def update_subset_aabbs(
    faces_idx: qd.types.ndarray(ndim=1),
    dyn_state: array_class.DynState,
    aabb_state: qd.template(),
    dyn_info: array_class.DynInfo,
    rigid_config: qd.template(),
):
    """Update the per-face collision AABBs of a per-env BVH covering the compacted face subset `faces_idx` (slot i_a
    holds face faces_idx[i_a]), so the rebuild scales with the subset size. See update_face_aabb."""
    for i_b, i_a in qd.ndrange(dyn_state.free_verts.pos.shape[1], faces_idx.shape[0]):
        update_face_aabb(i_b, i_a, faces_idx[i_a], i_b, dyn_state, aabb_state, dyn_info, rigid_config)


@qd.kernel
def kernel_update_verts_and_aabbs(
    dyn_state: array_class.DynState,
    aabb_state: qd.template(),
    dyn_info: array_class.DynInfo,
    rigid_config: qd.template(),
):
    func_update_all_verts(dyn_state, dyn_info, rigid_config)
    update_aabbs(dyn_state, aabb_state, dyn_info, rigid_config)


@qd.kernel
def kernel_update_verts_and_subset_aabbs(
    faces_idx: qd.types.ndarray(ndim=1),
    dyn_state: array_class.DynState,
    aabb_state: qd.template(),
    dyn_info: array_class.DynInfo,
    rigid_config: qd.template(),
):
    func_update_all_verts(dyn_state, dyn_info, rigid_config)
    update_subset_aabbs(faces_idx, dyn_state, aabb_state, dyn_info, rigid_config)


@qd.kernel
def kernel_update_subset_aabbs(
    faces_idx: qd.types.ndarray(ndim=1),
    dyn_state: array_class.DynState,
    aabb_state: qd.template(),
    dyn_info: array_class.DynInfo,
    rigid_config: qd.template(),
):
    update_subset_aabbs(faces_idx, dyn_state, aabb_state, dyn_info, rigid_config)


@qd.kernel
def kernel_update_grouped_aabbs(
    batch_repr_env: qd.types.ndarray(ndim=1),  # [n_trees] env whose geometry builds each tree slot
    dyn_state: array_class.DynState,
    aabb_state: qd.template(),
    dyn_info: array_class.DynInfo,
    rigid_config: qd.template(),
):
    """Build the per-face AABBs of a grouped static BVH covering every face in order: one tree slot per distinct
    per-env geometry, each built from its representative env (see RaycastContext update). The verts must be up to
    date (kernel_update_all_verts); grouping happens between the vert refresh and this build, which is why it is a
    separate kernel from kernel_update_verts_and_aabbs."""
    for i_t, i_f in qd.ndrange(aabb_state.aabbs.shape[0], dyn_info.faces.verts_idx.shape[0]):
        update_face_aabb(i_t, i_f, i_f, batch_repr_env[i_t], dyn_state, aabb_state, dyn_info, rigid_config)


@qd.kernel
def kernel_update_grouped_subset_aabbs(
    batch_repr_env: qd.types.ndarray(ndim=1),
    faces_idx: qd.types.ndarray(ndim=1),
    dyn_state: array_class.DynState,
    aabb_state: qd.template(),
    dyn_info: array_class.DynInfo,
    rigid_config: qd.template(),
):
    """Compacted-subset variant of kernel_update_grouped_aabbs; see update_subset_aabbs for faces_idx."""
    for i_t, i_a in qd.ndrange(aabb_state.aabbs.shape[0], faces_idx.shape[0]):
        update_face_aabb(i_t, i_a, faces_idx[i_a], batch_repr_env[i_t], dyn_state, aabb_state, dyn_info, rigid_config)


@qd.kernel
def kernel_remap_leaf_faces(faces_idx: qd.types.ndarray(ndim=1), bvh_morton_codes: qd.template()):
    """Rewrite the sorted leaf payloads of a compacted-subset BVH from subset slots to global face indices.

    Run once after each such build (build() recomputes the payloads); every traversal then reads global faces
    directly, with no per-leaf indirection and no knowledge of the subset. A zero-copy view write would be preferred
    for a pure accessor like this, but quadrants' DLPack export does not support u32 fields, so the kernel is the
    only implementation.
    """
    for i_b, i in qd.ndrange(bvh_morton_codes.shape[0], bvh_morton_codes.shape[1]):
        bvh_morton_codes[i_b, i][1] = qd.cast(faces_idx[qd.cast(bvh_morton_codes[i_b, i][1], gs.qd_int)], qd.u32)


# =========================================== Visual Mesh Raycasting ===========================================


@qd.func
def get_visual_vvert_pos(i_vv: int, i_b: int, dyn_state: array_class.DynState, dyn_info: array_class.DynInfo):
    """
    Return the world-space position of a visual vertex, branching between the custom buffer and FK on the fly.

    Opt-in entities (morph.enable_custom_vverts=True) own a slot in vverts_state.pos referenced by vverts_state_idx.
    Non-opt-in entities (vverts_state_idx<0) have no slot; their position is recomputed by transforming the rest-pose
    init_pos with the owning vgeom's current pose.
    """
    pos = qd.math.vec3(0.0, 0.0, 0.0)
    i_state = dyn_info.vverts.vverts_state_idx[i_vv]
    if i_state >= 0:
        pos = dyn_state.vverts.pos[i_state, i_b]
    else:
        i_vg = dyn_info.vverts.vgeom_idx[i_vv]
        pos = gu.qd_transform_by_trans_quat(
            dyn_info.vverts.init_pos[i_vv], dyn_state.vgeoms.pos[i_vg, i_b], dyn_state.vgeoms.quat[i_vg, i_b]
        )
    return pos


@qd.func
def get_visual_triangle_vertices(i_f: int, i_b: int, dyn_state: array_class.DynState, dyn_info: array_class.DynInfo):
    """Get the three vertices of a triangle from the visual mesh in world space."""
    tri_vertices = qd.Matrix.zero(gs.qd_float, 3, 3)
    for i in qd.static(range(3)):
        i_vv = dyn_info.vfaces.vverts_idx[i_f][i]
        tri_vertices[:, i] = get_visual_vvert_pos(i_vv, i_b, dyn_state, dyn_info)
    return tri_vertices


@qd.func
def bvh_ray_cast_visual(
    i_t,
    i_b,
    ray_start,
    ray_dir,
    max_range,
    bvh_nodes: qd.template(),
    bvh_morton_codes: qd.template(),
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    eps,
):
    """Cast a single ray against the visual-mesh BVH; returns (hit_face, distance, normal).

    See bvh_ray_cast for the tree slot (i_t) / env (i_b) split.
    """
    n_triangles = dyn_info.vfaces.vverts_idx.shape[0]

    hit_face = -1
    closest_distance = gs.qd_float(max_range)
    hit_normal = qd.math.vec3(0.0, 0.0, 0.0)

    axes, shear, is_valid_dir = ray_projection(ray_dir, eps)

    node_stack = qd.Vector.zero(gs.qd_int, qd.static(STACK_SIZE))
    node_stack[0] = 0
    stack_idx = 1
    if not is_valid_dir:
        stack_idx = 0

    while stack_idx > 0:
        stack_idx -= 1
        node_idx = node_stack[stack_idx]
        node = bvh_nodes[i_t, node_idx]

        aabb_t = ray_aabb_intersection(ray_start, ray_dir, node.bound.min, node.bound.max, eps)

        if aabb_t >= 0.0 and aabb_t < closest_distance:
            if node.left == -1:
                sorted_leaf_idx = node_idx - (n_triangles - 1)
                i_f = qd.cast(bvh_morton_codes[i_t, sorted_leaf_idx][1], gs.qd_int)

                tri_vertices = get_visual_triangle_vertices(i_f, i_b, dyn_state, dyn_info)
                v0, v1, v2 = tri_vertices[:, 0], tri_vertices[:, 1], tri_vertices[:, 2]

                hit_distance = ray_triangle_intersection(axes, ray_start, shear, v0, v1, v2, eps)

                if hit_distance >= 0.0 and hit_distance < closest_distance:
                    closest_distance = hit_distance
                    hit_face = i_f
                    hit_normal = triangle_face_normal(v0, v1, v2)
            else:
                if stack_idx < qd.static(STACK_SIZE - 2):
                    node_stack[stack_idx] = node.left
                    node_stack[stack_idx + 1] = node.right
                    stack_idx += 2

    return hit_face, closest_distance, hit_normal


@qd.func
def update_visual_face_aabb(
    i_t: int,
    i_f: int,
    i_b: int,
    face_mask: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    aabb_state: qd.template(),
    dyn_info: array_class.DynInfo,
    rigid_config: qd.template(),
):
    """Fit the AABB of vface i_f in tree slot i_t from env i_b's visual mesh (i_t == i_b for a per-env BVH).

    A vface that does not contribute to env i_b is left with an inverted AABB (min=+inf, max=-inf), which ray queries
    treat as a definitive miss: either face_mask holds 0, meaning its entity did not opt into raycasting, or its vgeom
    falls outside env i_b's active vgeom range (links_info.vgeom_start / vgeom_end). For a homogeneous solver every
    vgeom is always in range. For a heterogeneous solver, where all envs share one visual mesh but activate different
    per-env vgeom ranges, this keeps each env casting against its own variant alone.
    """
    aabb_state.aabbs[i_t, i_f].min.fill(qd.math.inf)
    aabb_state.aabbs[i_t, i_f].max.fill(-qd.math.inf)
    if face_mask[i_f] != 0:
        i_vg = dyn_info.vfaces.vgeom_idx[i_f]
        i_l = dyn_info.vgeoms.link_idx[i_vg]
        I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l
        if dyn_info.links.vgeom_start[I_l] <= i_vg and i_vg < dyn_info.links.vgeom_end[I_l]:
            for i in qd.static(range(3)):
                i_vv = dyn_info.vfaces.vverts_idx[i_f][i]
                pos_v = get_visual_vvert_pos(i_vv, i_b, dyn_state, dyn_info)
                aabb_state.aabbs[i_t, i_f].min = qd.min(aabb_state.aabbs[i_t, i_f].min, pos_v)
                aabb_state.aabbs[i_t, i_f].max = qd.max(aabb_state.aabbs[i_t, i_f].max, pos_v)


@qd.kernel
def kernel_update_visual_aabbs(
    face_mask: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    aabb_state: qd.template(),
    dyn_info: array_class.DynInfo,
    rigid_config: qd.template(),
):
    """Update the per-vface AABBs of a per-env visual BVH (one tree slot per env). See update_visual_face_aabb."""
    _B = dyn_state.vgeoms.pos.shape[1]
    n_vfaces = dyn_info.vfaces.vverts_idx.shape[0]
    for i_b, i_f in qd.ndrange(_B, n_vfaces):
        update_visual_face_aabb(i_b, i_f, i_b, face_mask, dyn_state, aabb_state, dyn_info, rigid_config)


@qd.kernel
def kernel_update_grouped_visual_aabbs(
    batch_repr_env: qd.types.ndarray(ndim=1),  # [n_trees] env whose geometry builds each tree slot
    face_mask: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    aabb_state: qd.template(),
    dyn_info: array_class.DynInfo,
    rigid_config: qd.template(),
):
    """Build the per-vface AABBs of a grouped static visual BVH: one tree slot per distinct per-env visual geometry,
    each built from its representative env (see RaycastContext update)."""
    for i_t, i_f in qd.ndrange(aabb_state.aabbs.shape[0], dyn_info.vfaces.vverts_idx.shape[0]):
        update_visual_face_aabb(i_t, i_f, batch_repr_env[i_t], face_mask, dyn_state, aabb_state, dyn_info, rigid_config)


# FIXME: Fastcache is not supported because of 'bvh_nodes', 'bvh_morton_codes'.
@qd.kernel(fastcache=False)
def kernel_cast_ray(
    ray_start: qd.types.ndarray(ndim=1),  # (3,)
    envs_idx: qd.types.ndarray(ndim=1),  # [n_envs]
    bvh_nodes: qd.template(),
    bvh_morton_codes: qd.template(),
    ray_direction: qd.types.ndarray(ndim=1),  # (3,)
    dyn_state: array_class.DynState,
    result: array_class.RaycastResult,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    max_range: float,
    eps: float,
    is_visual: qd.template(),
):
    """
    Cast a single ray against each env's BVH in parallel.

    Per-env: the ray is shifted by -envs_offset[i_b] (each BVH is in env-local coordinates) and the closest hit on
    that env is written to result[i_b]; envs not in envs_idx are left as no-hit (geom_idx == -1, distance == +inf).
    Aggregation across envs is intentionally out of scope, because cross-env reduction has no use beyond the viewer.

    `is_visual` selects the mesh the BVH covers: the visual mesh (vfaces, result.geom_idx holds the hit vgeom) or
    the collision mesh (faces, result.geom_idx holds the hit geom).
    """
    ray_start_world = qd.math.vec3(ray_start[0], ray_start[1], ray_start[2])
    ray_direction_world = qd.math.vec3(ray_direction[0], ray_direction[1], ray_direction[2])

    for i_b in range(result.geom_idx.shape[0]):
        result.distance[i_b] = qd.math.inf
        result.geom_idx[i_b] = -1
        result.hit_point[i_b] = qd.math.vec3(0.0, 0.0, 0.0)
        result.normal[i_b] = qd.math.vec3(0.0, 0.0, 0.0)

    for i_b_ in range(envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        env_offset = rigid_info.envs_offset[i_b]
        # Declared before the compile-time branch, whose body is a nested scope in quadrants.
        cur_hit_face = -1
        cur_distance = gs.qd_float(max_range)
        cur_hit_normal = qd.math.vec3(0.0, 0.0, 0.0)
        if qd.static(is_visual):
            cur_hit_face, cur_distance, cur_hit_normal = bvh_ray_cast_visual(
                i_b,
                i_b,
                ray_start_world - env_offset,
                ray_direction_world,
                max_range,
                bvh_nodes,
                bvh_morton_codes,
                dyn_state,
                dyn_info,
                eps,
            )
        else:
            # Single-ray (viewer) path: no per-sensor context, so i_s=0, an
            # empty link_excluded mask and is_link_excluded=False (no filtering).
            cur_hit_face, cur_distance, cur_hit_normal = bvh_ray_cast(
                i_b,
                i_b,
                0,
                ray_start_world - env_offset,
                ray_direction_world,
                max_range,
                bvh_nodes,
                bvh_morton_codes,
                None,
                dyn_state,
                dyn_info,
                eps,
                False,
            )

        if cur_hit_face >= 0:
            result.distance[i_b] = cur_distance
            if qd.static(is_visual):
                result.geom_idx[i_b] = dyn_info.vfaces.vgeom_idx[cur_hit_face]
            else:
                result.geom_idx[i_b] = dyn_info.faces.geom_idx[cur_hit_face]
            result.normal[i_b] = cur_hit_normal
            result.hit_point[i_b] = ray_start_world + cur_distance * ray_direction_world


@qd.func
def write_ray_hit(
    i_b: int,
    i_s: int,
    i_p_sensor: int,
    i_p_offset: int,
    i_p_dist: int,
    hit_face: int,
    hit_distance: float,
    ray_start_world,
    ray_direction_world,
    ray_dir_local,
    is_world_frame: qd.types.ndarray(ndim=1),
    max_ranges: qd.types.ndarray(ndim=1),
    no_hit_values: qd.types.ndarray(ndim=1),
    sensor_return_points: qd.types.ndarray(ndim=1),
    output_hits: qd.types.ndarray(ndim=2),
    eps: float,
    is_merge: qd.template(),
    is_last: qd.template(),
):
    """Common post-BVH write block for both collision and visual cast kernels.

    `is_merge` and `is_last` are compile-time flags marking a cast's position in a chain of BVH passes sharing one
    output buffer (first pass is_merge=False, subsequent passes is_merge=True, final pass is_last=True): the first
    pass writes a value into every slot, and each later pass only overwrites a slot when it found a closer hit, so
    the chain composes with no scratch storage. A miss must not beat a real hit from another pass whatever the
    sensor's ``no_hit_value`` (it may be below max_range, e.g. 0 or -1): a miss on a non-final pass seeds the slot
    with ``max_range`` - a hit is always strictly below it, so a hit wins every distance comparison - and
    ``no_hit_value`` is stamped only by the final pass, over any slot still holding that sentinel.

    `sensor_return_points[i_s]` gates the hit-point writes; a distances-only sensor skips them.
    """
    if hit_face >= 0 and (not is_merge or hit_distance < output_hits[i_p_dist, i_b]):
        output_hits[i_p_dist, i_b] = hit_distance

        if sensor_return_points[i_s]:
            hit_point = qd.math.vec3(0.0, 0.0, 0.0)
            if is_world_frame[i_s]:
                hit_point = ray_start_world + hit_distance * ray_direction_world
            else:
                # Local frame output along provided local ray direction
                hit_point = hit_distance * gu.qd_normalize(ray_dir_local, eps)
            # Store points at: cache_offset + point_idx_in_sensor * 3
            output_hits[i_p_offset + i_p_sensor * 3 + 0, i_b] = hit_point.x
            output_hits[i_p_offset + i_p_sensor * 3 + 1, i_b] = hit_point.y
            output_hits[i_p_offset + i_p_sensor * 3 + 2, i_b] = hit_point.z
    elif not is_merge:
        # First-pass miss: zero the point and seed the distance - no_hit_value if this pass is also the last (single
        # BVH), else the max_range sentinel so a later pass's hit wins.
        if sensor_return_points[i_s]:
            output_hits[i_p_offset + i_p_sensor * 3 + 0, i_b] = 0.0
            output_hits[i_p_offset + i_p_sensor * 3 + 1, i_b] = 0.0
            output_hits[i_p_offset + i_p_sensor * 3 + 2, i_b] = 0.0
        if is_last:
            output_hits[i_p_dist, i_b] = no_hit_values[i_s]
        else:
            output_hits[i_p_dist, i_b] = max_ranges[i_s]
    elif is_last:
        # Final-pass miss: a slot still at the sentinel means every pass missed, so stamp no_hit_value.
        if output_hits[i_p_dist, i_b] >= max_ranges[i_s]:
            output_hits[i_p_dist, i_b] = no_hit_values[i_s]


@qd.kernel
def kernel_cast_rays(
    points_to_sensor_idx: qd.types.ndarray(ndim=1),  # [n_points]
    env_bvh_idx: qd.types.ndarray(ndim=1),  # [n_env] - BVH tree slot each env casts against
    bvh_nodes: qd.template(),
    bvh_morton_codes: qd.template(),  # maps sorted leaves to original triangle indices
    links_pos: qd.types.ndarray(ndim=3),  # [n_env, n_sensors, 3]
    links_quat: qd.types.ndarray(ndim=3),  # [n_env, n_sensors, 4]
    ray_starts: qd.types.ndarray(ndim=2),  # [n_points, 3]
    ray_directions: qd.types.ndarray(ndim=2),  # [n_points, 3]
    max_ranges: qd.types.ndarray(ndim=1),  # [n_sensors]
    no_hit_values: qd.types.ndarray(ndim=1),  # [n_sensors]
    is_world_frame: qd.types.ndarray(ndim=1),  # [n_sensors]
    sensor_cache_offsets: qd.types.ndarray(ndim=1),  # [n_sensors] - cache start index for each sensor
    sensor_point_offsets: qd.types.ndarray(ndim=1),  # [n_sensors] - point start index for each sensor
    sensor_point_counts: qd.types.ndarray(ndim=1),  # [n_sensors] - number of points for each sensor
    sensor_return_points: qd.types.ndarray(ndim=1),  # [n_sensors] - True to store hit points, False for distances-only
    link_excluded: qd.types.ndarray(ndim=2),  # [n_sensors, n_links] bool - 0 marks an excluded link's faces
    output_hits: qd.types.ndarray(ndim=2),  # [total_cache_size, n_env]
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    eps: float,
    is_merge: qd.template(),
    is_last: qd.template(),
    is_env_major: qd.template(),
    is_link_excluded: qd.template(),
):
    """Cast rays against a collision-mesh BVH.

    See write_ray_hit for `is_merge` / `is_last` semantics. The result `output_hits` is a 2D array of shape
    (total_cache_size, n_env) where in the first dimension each sensor's data is stored as [sensor_points
    (n_points * 3), sensor_ranges (n_points)], the point block being present only for sensors with
    sensor_return_points set.

    env_bvh_idx routes each env to the BVH tree slot it casts against: identity for per-env trees, all-zero for one
    shared tree, group ids for a grouped static BVH (N distinct geometries, N <= n_env). is_env_major is a
    compile-time flag selecting the thread -> (ray, env) mapping matching that tree layout.
    """
    n_points = ray_starts.shape[0]
    n_envs = output_hits.shape[-1]
    # One flat parallel loop whose thread -> (ray, env) split is chosen at compile time from is_env_major:
    #  - env-major (one tree serves several envs): env is the fastest-varying index, so a warp spans consecutive
    #    envs mostly reading the same tree's nodes -> a coalesced broadcast.
    #  - ray-major (distinct per-env trees): the ray is the fastest-varying index, so a warp stays within one env's
    #    tree and rides ray coherence instead of diverging across n_env different trees.
    for i_flat in range(n_points * n_envs):
        i_p = i_flat // n_envs
        i_b = i_flat % n_envs
        if not is_env_major:
            i_b = i_flat // n_points
            i_p = i_flat % n_points

        i_s = points_to_sensor_idx[i_p]

        link_pos = qd.math.vec3(links_pos[i_b, i_s, 0], links_pos[i_b, i_s, 1], links_pos[i_b, i_s, 2])
        link_quat = qd.math.vec4(
            links_quat[i_b, i_s, 0], links_quat[i_b, i_s, 1], links_quat[i_b, i_s, 2], links_quat[i_b, i_s, 3]
        )

        ray_start_local = qd.math.vec3(ray_starts[i_p, 0], ray_starts[i_p, 1], ray_starts[i_p, 2])
        ray_start_world = gu.qd_transform_by_trans_quat(ray_start_local, link_pos, link_quat)

        ray_dir_local = qd.math.vec3(ray_directions[i_p, 0], ray_directions[i_p, 1], ray_directions[i_p, 2])
        ray_direction_world = gu.qd_normalize(gu.qd_transform_by_quat(ray_dir_local, link_quat), eps)

        hit_face, hit_distance, _hit_normal = bvh_ray_cast(
            env_bvh_idx[i_b],
            i_b,
            i_s,
            ray_start_world,
            ray_direction_world,
            max_ranges[i_s],
            bvh_nodes,
            bvh_morton_codes,
            link_excluded,
            dyn_state,
            dyn_info,
            eps,
            is_link_excluded,
        )

        i_p_sensor = i_p - sensor_point_offsets[i_s]
        i_p_offset = sensor_cache_offsets[i_s]
        # Distances follow the point block (num_rays*3) when points are stored, else start at the block front.
        i_p_dist = i_p_offset + i_p_sensor
        if sensor_return_points[i_s]:
            i_p_dist += sensor_point_counts[i_s] * 3
        write_ray_hit(
            i_b,
            i_s,
            i_p_sensor,
            i_p_offset,
            i_p_dist,
            hit_face,
            hit_distance,
            ray_start_world,
            ray_direction_world,
            ray_dir_local,
            is_world_frame,
            max_ranges,
            no_hit_values,
            sensor_return_points,
            output_hits,
            eps,
            is_merge,
            is_last,
        )


@qd.kernel
def kernel_cast_rays_visual(
    points_to_sensor_idx: qd.types.ndarray(ndim=1),
    env_bvh_idx: qd.types.ndarray(ndim=1),
    bvh_nodes: qd.template(),
    bvh_morton_codes: qd.template(),
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
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    eps: float,
    is_merge: qd.template(),
    is_last: qd.template(),
    is_env_major: qd.template(),
):
    """Visual-mesh variant of kernel_cast_rays.

    See kernel_cast_rays for env_bvh_idx, is_env_major and the thread mapping.
    """
    n_points = ray_starts.shape[0]
    n_envs = output_hits.shape[-1]
    for i_flat in range(n_points * n_envs):
        i_p = i_flat // n_envs
        i_b = i_flat % n_envs
        if not is_env_major:
            i_b = i_flat // n_points
            i_p = i_flat % n_points

        i_s = points_to_sensor_idx[i_p]

        link_pos = qd.math.vec3(links_pos[i_b, i_s, 0], links_pos[i_b, i_s, 1], links_pos[i_b, i_s, 2])
        link_quat = qd.math.vec4(
            links_quat[i_b, i_s, 0], links_quat[i_b, i_s, 1], links_quat[i_b, i_s, 2], links_quat[i_b, i_s, 3]
        )

        ray_start_local = qd.math.vec3(ray_starts[i_p, 0], ray_starts[i_p, 1], ray_starts[i_p, 2])
        ray_start_world = gu.qd_transform_by_trans_quat(ray_start_local, link_pos, link_quat)

        ray_dir_local = qd.math.vec3(ray_directions[i_p, 0], ray_directions[i_p, 1], ray_directions[i_p, 2])
        ray_direction_world = gu.qd_normalize(gu.qd_transform_by_quat(ray_dir_local, link_quat), eps)

        hit_face, hit_distance, _hit_normal = bvh_ray_cast_visual(
            env_bvh_idx[i_b],
            i_b,
            ray_start_world,
            ray_direction_world,
            max_ranges[i_s],
            bvh_nodes,
            bvh_morton_codes,
            dyn_state,
            dyn_info,
            eps,
        )

        i_p_sensor = i_p - sensor_point_offsets[i_s]
        i_p_offset = sensor_cache_offsets[i_s]
        # Distances follow the point block (num_rays*3) when points are stored, else start at the block front.
        i_p_dist = i_p_offset + i_p_sensor
        if sensor_return_points[i_s]:
            i_p_dist += sensor_point_counts[i_s] * 3
        write_ray_hit(
            i_b,
            i_s,
            i_p_sensor,
            i_p_offset,
            i_p_dist,
            hit_face,
            hit_distance,
            ray_start_world,
            ray_direction_world,
            ray_dir_local,
            is_world_frame,
            max_ranges,
            no_hit_values,
            sensor_return_points,
            output_hits,
            eps,
            is_merge,
            is_last,
        )
