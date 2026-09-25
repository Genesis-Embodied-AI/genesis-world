"""
Expanding Polytope Algorithm (EPA) for penetration depth computation.

This module contains the EPA algorithm implementation for computing exact
penetration depth and contact normals for intersecting convex objects.
Includes both standard and numerically robust ("safe") variants.
"""

import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
import genesis.utils.array_class as array_class

from .constants import RETURN_CODE, EPA_POLY_INIT_RETURN_CODE, GJK_RETURN_CODE
from .gjk_utils import (
    func_triangle_affine_coords,
    func_ray_triangle_intersection,
    func_point_triangle_intersection,
    func_origin_tetra_intersection,
    func_project_origin_to_plane,
)
from .utils import func_is_discrete_geoms

# Import func_support from gjk_support to avoid circular dependency
from .gjk_support import func_support


@qd.func
def func_epa(
    i_ga,
    i_gb,
    i_b,
    pos_a: qd.types.vector(3),
    quat_a: qd.types.vector(4),
    pos_b: qd.types.vector(3),
    quat_b: qd.types.vector(4),
    collider_state: array_class.ColliderState,
    gjk_state: array_class.GJKState,
    dyn_info: array_class.DynInfo,
    collider_info: array_class.ColliderInfo,
    rigid_config: qd.template(),
    collider_static_config: qd.template(),
):
    """
    EPA algorithm to find the exact penetration depth and contact normal using the simplex constructed by GJK.

    .. seealso::
    MuJoCo's original implementation:
    https://github.com/google-deepmind/mujoco/blob/7dc7a349c5ba2db2d3f8ab50a367d08e2f1afbbc/src/engine/engine_collision_gjk.c#L1331
    """
    upper = collider_info.gjk.FLOAT_MAX[None]
    upper2 = collider_info.gjk.FLOAT_MAX_SQ[None]
    lower = 0.0
    tolerance = collider_info.gjk.tolerance[None]

    # Index of the nearest face
    nearest_i_f = -1
    prev_nearest_i_f = -1

    discrete = func_is_discrete_geoms(i_ga, i_gb, dyn_info)
    if discrete:
        # If the objects are discrete, we do not use tolerance.
        tolerance = collider_info.gjk.FLOAT_MIN[None]

    k_max = collider_info.gjk.epa_max_iterations[None]
    for k in range(k_max):
        prev_nearest_i_f = nearest_i_f

        # Find the polytope face with the smallest distance to the origin
        lower2 = collider_info.gjk.FLOAT_MAX_SQ[None]

        for i in range(gjk_state.polytope.nfaces_map[i_b]):
            i_f = gjk_state.polytope_faces_map[i_b, i]
            face_dist2 = gjk_state.polytope_faces.dist2[i_b, i_f]

            if face_dist2 < lower2:
                lower2 = face_dist2
                nearest_i_f = i_f

        if lower2 > upper2 or nearest_i_f < 0:
            # Invalid face found, stop the algorithm (lower bound of depth is larger than upper bound)
            nearest_i_f = prev_nearest_i_f
            break

        if lower2 <= collider_info.gjk.FLOAT_MIN_SQ[None]:
            # Invalid lower bound (0), stop the algorithm (origin is on the affine hull of face)
            break

        # Find a new support point w from the nearest face's normal
        lower = qd.sqrt(lower2)
        dir = gjk_state.polytope_faces.normal[i_b, nearest_i_f]
        wi = func_epa_support(
            i_ga,
            i_gb,
            i_b,
            pos_a,
            quat_a,
            pos_b,
            quat_b,
            dir,
            lower,
            collider_state,
            gjk_state,
            dyn_info,
            collider_info,
            rigid_config,
            collider_static_config,
        )
        w = gjk_state.polytope_verts.mink[i_b, wi]

        # The upper bound of depth at k-th iteration
        upper_k = w.dot(dir) / lower
        if upper_k < upper:
            upper = upper_k
            upper2 = upper**2

        # If the upper bound and lower bound are close enough, we can stop the algorithm
        if (upper - lower) < tolerance:
            break

        if discrete:
            repeated = False
            for i in range(gjk_state.polytope.nverts[i_b] - 1):
                if (
                    gjk_state.polytope_verts.id1[i_b, i] == gjk_state.polytope_verts.id1[i_b, wi]
                    and gjk_state.polytope_verts.id2[i_b, i] == gjk_state.polytope_verts.id2[i_b, wi]
                ):
                    # The vertex w is already in the polytope,
                    # so we do not need to add it again.
                    repeated = True
                    break
            if repeated:
                break

        gjk_state.polytope.horizon_w[i_b] = w

        # Compute horizon
        horizon_flag = func_epa_horizon(i_b, nearest_i_f, gjk_state, collider_info)

        if horizon_flag:
            # There was an error in the horizon construction, so the horizon edge is not a closed loop.
            nearest_i_f = -1
            break

        if gjk_state.polytope.horizon_nedges[i_b] < 3:
            # Should not happen, because at least three edges should be in the horizon from one deleted face.
            nearest_i_f = -1
            break

        # Check if the memory space is enough for attaching new faces
        nfaces = gjk_state.polytope.nfaces[i_b]
        nedges = gjk_state.polytope.horizon_nedges[i_b]
        if nfaces + nedges >= collider_info.gjk.polytope_max_faces[None]:
            # If the polytope is full, we cannot insert new faces
            break

        # Attach the new faces
        for i in range(nedges):
            # Face id of the current face to attach
            i_f0 = nfaces + i
            # Face id of the next face to attach
            i_f1 = nfaces + (i + 1) % nedges

            horizon_i_f = gjk_state.polytope_horizon_data.face_idx[i_b, i]
            horizon_i_e = gjk_state.polytope_horizon_data.edge_idx[i_b, i]

            horizon_v1 = gjk_state.polytope_faces.verts_idx[i_b, horizon_i_f][horizon_i_e]
            horizon_v2 = gjk_state.polytope_faces.verts_idx[i_b, horizon_i_f][(horizon_i_e + 1) % 3]

            # Change the adjacent face index of the existing face
            gjk_state.polytope_faces.adj_idx[i_b, horizon_i_f][horizon_i_e] = i_f0

            # Attach the new face.
            # If this if the first face, will be adjacent to the face that will be attached last.
            adj_i_f_0 = i_f0 - 1 if (i > 0) else nfaces + nedges - 1
            adj_i_f_1 = horizon_i_f
            adj_i_f_2 = i_f1

            dist2 = func_attach_face_to_polytope(
                i_b,
                wi,
                horizon_v2,
                horizon_v1,
                adj_i_f_2,  # Previous face id
                adj_i_f_1,
                adj_i_f_0,  # Next face id
                gjk_state,
                collider_info,
            )
            if dist2 <= 0:
                # Unrecoverable numerical issue
                nearest_i_f = -1
                break

            if (dist2 >= lower2) and (dist2 <= upper2):
                # Store face in the map
                nfaces_map = gjk_state.polytope.nfaces_map[i_b]
                gjk_state.polytope_faces_map[i_b, nfaces_map] = i_f0
                gjk_state.polytope_faces.map_idx[i_b, i_f0] = nfaces_map
                gjk_state.polytope.nfaces_map[i_b] += 1

        # Clear the horizon data for the next iteration
        gjk_state.polytope.horizon_nedges[i_b] = 0

        if (gjk_state.polytope.nfaces_map[i_b] == 0) or (nearest_i_f == -1):
            # No face candidate left
            break

    if nearest_i_f != -1:
        # Nearest face found
        dist2 = gjk_state.polytope_faces.dist2[i_b, nearest_i_f]
        func_epa_witness(i_ga, i_gb, i_b, nearest_i_f, gjk_state)
        gjk_state.n_witness[i_b] = 1
        gjk_state.distance[i_b] = -qd.sqrt(dist2)
    else:
        # No face found, so the objects are not colliding
        gjk_state.n_witness[i_b] = 0
        gjk_state.distance[i_b] = 0

    return nearest_i_f


@qd.func
def func_epa_witness(i_ga, i_gb, i_b, i_f, gjk_state: array_class.GJKState):
    """
    Compute the witness points from the geometries for the face i_f of the polytope.
    """
    # Find the affine coordinates of the origin's projection on the face i_f
    face_iv1 = gjk_state.polytope_faces.verts_idx[i_b, i_f][0]
    face_iv2 = gjk_state.polytope_faces.verts_idx[i_b, i_f][1]
    face_iv3 = gjk_state.polytope_faces.verts_idx[i_b, i_f][2]
    face_v1 = gjk_state.polytope_verts.mink[i_b, face_iv1]
    face_v2 = gjk_state.polytope_verts.mink[i_b, face_iv2]
    face_v3 = gjk_state.polytope_verts.mink[i_b, face_iv3]
    face_normal = gjk_state.polytope_faces.normal[i_b, i_f]

    _lambda = func_triangle_affine_coords(face_normal, face_v1, face_v2, face_v3)

    # Point on geom 1
    v1 = gjk_state.polytope_verts.obj1[i_b, face_iv1]
    v2 = gjk_state.polytope_verts.obj1[i_b, face_iv2]
    v3 = gjk_state.polytope_verts.obj1[i_b, face_iv3]
    witness1 = v1 * _lambda[0] + v2 * _lambda[1] + v3 * _lambda[2]

    # Point on geom 2
    v1 = gjk_state.polytope_verts.obj2[i_b, face_iv1]
    v2 = gjk_state.polytope_verts.obj2[i_b, face_iv2]
    v3 = gjk_state.polytope_verts.obj2[i_b, face_iv3]
    witness2 = v1 * _lambda[0] + v2 * _lambda[1] + v3 * _lambda[2]

    gjk_state.witness.point_obj1[i_b, 0] = witness1
    gjk_state.witness.point_obj2[i_b, 0] = witness2


@qd.func
def func_epa_horizon(i_b, nearest_i_f, gjk_state: array_class.GJKState, collider_info: array_class.ColliderInfo):
    """
    Compute the horizon, which represents the area of the polytope that is visible from the vertex w, and thus
    should be deleted for the expansion of the polytope.
    """
    w = gjk_state.polytope.horizon_w[i_b]

    # Initialize the stack by inserting the nearest face
    gjk_state.polytope_horizon_stack.face_idx[i_b, 0] = nearest_i_f
    gjk_state.polytope_horizon_stack.edge_idx[i_b, 0] = 0
    top = 1
    is_first = True

    flag = RETURN_CODE.SUCCESS
    while top > 0:
        # Pop the top face from the stack
        i_f = gjk_state.polytope_horizon_stack.face_idx[i_b, top - 1]
        i_e = gjk_state.polytope_horizon_stack.edge_idx[i_b, top - 1]
        i_v = gjk_state.polytope_faces.verts_idx[i_b, i_f][0]
        v = gjk_state.polytope_verts.mink[i_b, i_v]
        top -= 1

        # If the face is already deleted, skip it
        is_deleted = gjk_state.polytope_faces.map_idx[i_b, i_f] == -2
        if (not is_first) and (is_deleted):
            continue

        # Check visibility of the face. Two requirements for the face to be visible:
        # 1. The face normal should point towards the vertex w
        # 2. The vertex w should be on the other side of the face to the origin
        is_visible = gjk_state.polytope_faces.normal[i_b, i_f].dot(w - v) > collider_info.gjk.FLOAT_MIN[None]

        # The first face is always considered visible.
        if is_visible or is_first:
            # If visible, delete the face from the polytope
            func_delete_face_from_polytope(i_b, i_f, gjk_state)

            # Add the other two or three edges of the face to the stack.
            # The order is important to form a closed loop.
            for k in range(0 if is_first else 1, 3):
                i_e2 = (i_e + k) % 3
                adj_face_idx = gjk_state.polytope_faces.adj_idx[i_b, i_f][i_e2]
                adj_face_is_deleted = gjk_state.polytope_faces.map_idx[i_b, adj_face_idx] == -2
                if not adj_face_is_deleted:
                    # Get the related edge id from the adjacent face. Since adjacent faces have different
                    # orientations, we need to use the ending vertex of the edge.
                    start_vert_idx = gjk_state.polytope_faces.verts_idx[i_b, i_f][(i_e2 + 1) % 3]
                    adj_edge_idx = func_get_edge_idx(i_b, adj_face_idx, start_vert_idx, gjk_state)

                    gjk_state.polytope_horizon_stack.face_idx[i_b, top] = adj_face_idx
                    gjk_state.polytope_horizon_stack.edge_idx[i_b, top] = adj_edge_idx
                    top += 1
        else:
            # If not visible, add the edge to the horizon.
            flag = func_add_edge_to_horizon(i_b, i_f, i_e, gjk_state)
            if flag:
                # If the edges do not form a closed loop, there is an error in the algorithm.
                break

        is_first = False

    return flag


@qd.func
def func_add_edge_to_horizon(i_b, i_f, i_e, gjk_state: array_class.GJKState):
    """
    Add an edge to the horizon data structure.
    """
    horizon_nedges = gjk_state.polytope.horizon_nedges[i_b]
    gjk_state.polytope_horizon_data.edge_idx[i_b, horizon_nedges] = i_e
    gjk_state.polytope_horizon_data.face_idx[i_b, horizon_nedges] = i_f
    gjk_state.polytope.horizon_nedges[i_b] += 1

    return RETURN_CODE.SUCCESS


@qd.func
def func_get_edge_idx(i_b, i_f, i_v, gjk_state: array_class.GJKState):
    """
    Get the edge index from the face, starting from the vertex i_v.

    If the face is comprised of [v1, v2, v3], the edges are: [v1, v2], [v2, v3], [v3, v1].
    Therefore, if i_v was v1, the edge index is 0, and if i_v was v2, the edge index is 1.
    """
    verts = gjk_state.polytope_faces.verts_idx[i_b, i_f]
    ret = gs.qd_int(2)
    if verts[0] == i_v:
        ret = 0
    elif verts[1] == i_v:
        ret = 1
    return ret


@qd.func
def func_delete_face_from_polytope(i_b, i_f, gjk_state: array_class.GJKState):
    """
    Delete the face from the polytope.
    """
    face_map_idx = gjk_state.polytope_faces.map_idx[i_b, i_f]
    if face_map_idx >= 0:
        last_face_idx = gjk_state.polytope_faces_map[i_b, gjk_state.polytope.nfaces_map[i_b] - 1]
        # Make the map to point to the last face
        gjk_state.polytope_faces_map[i_b, face_map_idx] = last_face_idx
        # Change map index of the last face
        gjk_state.polytope_faces.map_idx[i_b, last_face_idx] = face_map_idx

        # Decrease the number of faces in the polytope
        gjk_state.polytope.nfaces_map[i_b] -= 1

    # Mark the face as deleted
    gjk_state.polytope_faces.map_idx[i_b, i_f] = -2


@qd.func
def func_epa_insert_vertex_to_polytope(
    i_b: int,
    obj1_id: int,
    obj2_id: int,
    obj1_point,
    obj2_point,
    obj1_localpos,
    obj2_localpos,
    minkowski_point,
    gjk_state: array_class.GJKState,
):
    """
    Copy vertex information into the polytope.
    """
    n = gjk_state.polytope.nverts[i_b]
    gjk_state.polytope_verts.obj1[i_b, n] = obj1_point
    gjk_state.polytope_verts.obj2[i_b, n] = obj2_point
    gjk_state.polytope_verts.local_obj1[i_b, n] = obj1_localpos
    gjk_state.polytope_verts.local_obj2[i_b, n] = obj2_localpos
    gjk_state.polytope_verts.id1[i_b, n] = obj1_id
    gjk_state.polytope_verts.id2[i_b, n] = obj2_id
    gjk_state.polytope_verts.mink[i_b, n] = minkowski_point
    gjk_state.polytope.nverts[i_b] += 1
    return n


@qd.func
def func_epa_init_polytope_2d(
    i_ga,
    i_gb,
    i_b,
    pos_a: qd.types.vector(3),
    quat_a: qd.types.vector(4),
    pos_b: qd.types.vector(3),
    quat_b: qd.types.vector(4),
    collider_state: array_class.ColliderState,
    gjk_state: array_class.GJKState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    collider_info: array_class.ColliderInfo,
    rigid_config: qd.template(),
    collider_static_config: qd.template(),
):
    """
    Create the polytope for EPA from a 1-simplex (line segment).

    Returns
    -------
    int
        0 when successful, or a flag indicating an error.
    """
    flag = EPA_POLY_INIT_RETURN_CODE.SUCCESS

    # Get the simplex vertices
    v1 = gjk_state.simplex_vertex.mink[i_b, 0]
    v2 = gjk_state.simplex_vertex.mink[i_b, 1]
    diff = v2 - v1

    # Find the element in [diff] with the smallest magnitude, because it will give us the largest cross product
    min_val = qd.abs(diff[0])
    min_i = 0
    for i in qd.static(range(1, 3)):
        abs_diff_i = qd.abs(diff[i])
        if abs_diff_i < min_val:
            min_val = abs_diff_i
            min_i = i

    # Cross product with the found axis, then rotate it by 120 degrees around the axis [diff] to get three more
    # points spaced 120 degrees apart
    rotmat = gu.qd_rotvec_to_R(diff * qd.math.radians(120.0), rigid_info.EPS[None])
    e = gs.qd_vec3(0.0, 0.0, 0.0)
    e[min_i] = 1.0

    d1 = e.cross(diff)
    d2 = rotmat @ d1
    d3 = rotmat @ d2

    # Insert the first two vertices into the polytope
    vi = qd.Vector([0, 0, 0, 0, 0], dt=qd.i32)
    for i in range(2):
        vi[i] = func_epa_insert_vertex_to_polytope(
            i_b,
            gjk_state.simplex_vertex.id1[i_b, i],
            gjk_state.simplex_vertex.id2[i_b, i],
            gjk_state.simplex_vertex.obj1[i_b, i],
            gjk_state.simplex_vertex.obj2[i_b, i],
            gjk_state.simplex_vertex.local_obj1[i_b, i],
            gjk_state.simplex_vertex.local_obj2[i_b, i],
            gjk_state.simplex_vertex.mink[i_b, i],
            gjk_state,
        )

    # Find three more vertices using [d1, d2, d3] as support vectors, and insert them into the polytope
    for i in range(3):
        di = d1
        if i == 1:
            di = d2
        elif i == 2:
            di = d3
        di_norm = di.norm()
        vi[i + 2] = func_epa_support(
            i_ga,
            i_gb,
            i_b,
            pos_a,
            quat_a,
            pos_b,
            quat_b,
            di,
            di_norm,
            collider_state,
            gjk_state,
            dyn_info,
            collider_info,
            rigid_config,
            collider_static_config,
        )

    v3 = gjk_state.polytope_verts.mink[i_b, vi[2]]
    v4 = gjk_state.polytope_verts.mink[i_b, vi[3]]
    v5 = gjk_state.polytope_verts.mink[i_b, vi[4]]

    # Build hexahedron (6 faces) from the five vertices.
    # * This hexahedron would have line [v1, v2] as the central axis, and the other three vertices would be on the
    # sides of the hexahedron, as they are spaced 120 degrees apart.
    # * We already know the face and adjacent face indices in building this.
    # * While building the hexahedron by attaching faces, if the face is very close to the origin, we replace the
    # 1-simplex with the 2-simplex, and restart from it.
    for i in range(6):
        # Vertex indices for the faces in the hexahedron
        i_v1, i_v2, i_v3 = vi[0], vi[2], vi[3]
        # Adjacent face indices for the faces in the hexahedron
        i_a1, i_a2, i_a3 = 1, 3, 2
        if i == 1:
            i_v1, i_v2, i_v3 = vi[0], vi[4], vi[2]
            i_a1, i_a2, i_a3 = 2, 4, 0
        elif i == 2:
            i_v1, i_v2, i_v3 = vi[0], vi[3], vi[4]
            i_a1, i_a2, i_a3 = 0, 5, 1
        elif i == 3:
            i_v1, i_v2, i_v3 = vi[1], vi[3], vi[2]
            i_a1, i_a2, i_a3 = 5, 0, 4
        elif i == 4:
            i_v1, i_v2, i_v3 = vi[1], vi[2], vi[4]
            i_a1, i_a2, i_a3 = 3, 1, 5
        elif i == 5:
            i_v1, i_v2, i_v3 = vi[1], vi[4], vi[3]
            i_a1, i_a2, i_a3 = 4, 2, 3

        if (
            func_attach_face_to_polytope(i_b, i_v1, i_v2, i_v3, i_a1, i_a2, i_a3, gjk_state, collider_info)
            < collider_info.gjk.FLOAT_MIN_SQ[None]
        ):
            func_replace_simplex_3(i_b, i_v1, i_v2, i_v3, gjk_state)
            flag = EPA_POLY_INIT_RETURN_CODE.P2_FALLBACK3
            break

    if flag == RETURN_CODE.SUCCESS:
        if not func_ray_triangle_intersection(v1, v2, v3, v4, v5):
            # The hexahedron should be convex by definition, but somehow if it is not, we return non-convex flag
            flag = EPA_POLY_INIT_RETURN_CODE.P2_NONCONVEX

    if flag == RETURN_CODE.SUCCESS:
        # Initialize face map
        for i in qd.static(range(6)):
            gjk_state.polytope_faces_map[i_b, i] = i
            gjk_state.polytope_faces.map_idx[i_b, i] = i
        gjk_state.polytope.nfaces_map[i_b] = 6

    return flag


@qd.func
def func_epa_init_polytope_3d(
    i_ga,
    i_gb,
    i_b,
    pos_a: qd.types.vector(3),
    quat_a: qd.types.vector(4),
    pos_b: qd.types.vector(3),
    quat_b: qd.types.vector(4),
    collider_state: array_class.ColliderState,
    gjk_state: array_class.GJKState,
    dyn_info: array_class.DynInfo,
    collider_info: array_class.ColliderInfo,
    rigid_config: qd.template(),
    collider_static_config: qd.template(),
):
    """
    Create the polytope for EPA from a 2-simplex (triangle).

    Returns
    -------
    int
        0 when successful, or a flag indicating an error.
    """
    flag = EPA_POLY_INIT_RETURN_CODE.SUCCESS

    # Get the simplex vertices
    v1 = gjk_state.simplex_vertex.mink[i_b, 0]
    v2 = gjk_state.simplex_vertex.mink[i_b, 1]
    v3 = gjk_state.simplex_vertex.mink[i_b, 2]

    # Get normal; if it is zero, we cannot proceed
    n = (v2 - v1).cross(v3 - v1)
    n_norm = n.norm()
    if n_norm < collider_info.gjk.FLOAT_MIN[None]:
        flag = EPA_POLY_INIT_RETURN_CODE.P3_BAD_NORMAL
    n_neg = -n

    # Save vertices in the polytope
    vi = qd.Vector([0, 0, 0, 0, 0], dt=qd.i32)
    for i in range(3):
        vi[i] = func_epa_insert_vertex_to_polytope(
            i_b,
            gjk_state.simplex_vertex.id1[i_b, i],
            gjk_state.simplex_vertex.id2[i_b, i],
            gjk_state.simplex_vertex.obj1[i_b, i],
            gjk_state.simplex_vertex.obj2[i_b, i],
            gjk_state.simplex_vertex.local_obj1[i_b, i],
            gjk_state.simplex_vertex.local_obj2[i_b, i],
            gjk_state.simplex_vertex.mink[i_b, i],
            gjk_state,
        )

    # Find the fourth and fifth vertices using the normal
    # as the support vector. We form a hexahedron (6 faces)
    # with these five vertices.
    for i in range(2):
        dir = n if i == 0 else n_neg
        vi[i + 3] = func_epa_support(
            i_ga,
            i_gb,
            i_b,
            pos_a,
            quat_a,
            pos_b,
            quat_b,
            dir,
            n_norm,
            collider_state,
            gjk_state,
            dyn_info,
            collider_info,
            rigid_config,
            collider_static_config,
        )
    v4 = gjk_state.polytope_verts.mink[i_b, vi[3]]
    v5 = gjk_state.polytope_verts.mink[i_b, vi[4]]

    # Check if v4 or v5 located inside the triangle.
    # If so, we do not proceed anymore.
    for i in range(2):
        v = v4 if i == 0 else v5
        if func_point_triangle_intersection(v, v1, v2, v3, collider_info):
            flag = EPA_POLY_INIT_RETURN_CODE.P3_INVALID_V4 if i == 0 else EPA_POLY_INIT_RETURN_CODE.P3_INVALID_V5
            break

    if flag == EPA_POLY_INIT_RETURN_CODE.SUCCESS:
        # If origin does not lie inside the triangle, we need to
        # check if the hexahedron contains the origin.

        tets_has_origin = gs.qd_ivec2(0, 0)
        for i in range(2):
            v = v4 if i == 0 else v5
            tets_has_origin[i] = 1 if func_origin_tetra_intersection(v1, v2, v3, v) == RETURN_CODE.SUCCESS else 0

        # @TODO: It's possible for GJK to return a triangle with origin not contained in it but within tolerance
        # from it. In that case, the hexahedron could possibly be constructed that does ont contain the origin, but
        # there is penetration depth.
        # The check only runs when GJK reported a positive separation: a reported penetration means the origin is
        # enclosed up to rounding, where the containment test is not trustworthy.
        if (
            gjk_state.distance[i_b] > 10 * collider_info.gjk.FLOAT_MIN[None]
            and (not tets_has_origin[0])
            and (not tets_has_origin[1])
        ):
            flag = EPA_POLY_INIT_RETURN_CODE.P3_MISSING_ORIGIN
        else:
            # Build hexahedron (6 faces) from the five vertices.
            for i in range(6):
                # Vertex indices for the faces in the hexahedron
                i_v1, i_v2, i_v3 = vi[3], vi[0], vi[1]
                # Adjacent face indices for the faces in the hexahedron
                i_a1, i_a2, i_a3 = 1, 3, 2
                if i == 1:
                    i_v1, i_v2, i_v3 = vi[3], vi[2], vi[0]
                    i_a1, i_a2, i_a3 = 2, 4, 0
                elif i == 2:
                    i_v1, i_v2, i_v3 = vi[3], vi[1], vi[2]
                    i_a1, i_a2, i_a3 = 0, 5, 1
                elif i == 3:
                    i_v1, i_v2, i_v3 = vi[4], vi[1], vi[0]
                    i_a1, i_a2, i_a3 = 5, 0, 4
                elif i == 4:
                    i_v1, i_v2, i_v3 = vi[4], vi[0], vi[2]
                    i_a1, i_a2, i_a3 = 3, 1, 5
                elif i == 5:
                    i_v1, i_v2, i_v3 = vi[4], vi[2], vi[1]
                    i_a1, i_a2, i_a3 = 4, 2, 3

                dist2 = func_attach_face_to_polytope(i_b, i_v1, i_v2, i_v3, i_a1, i_a2, i_a3, gjk_state, collider_info)
                if dist2 < collider_info.gjk.FLOAT_MIN_SQ[None]:
                    flag = EPA_POLY_INIT_RETURN_CODE.P3_ORIGIN_ON_FACE
                    break

    if flag == EPA_POLY_INIT_RETURN_CODE.SUCCESS:
        # Initialize face map
        for i in qd.static(range(6)):
            gjk_state.polytope_faces_map[i_b, i] = i
            gjk_state.polytope_faces.map_idx[i_b, i] = i
        gjk_state.polytope.nfaces_map[i_b] = 6

    return flag


@qd.func
def func_epa_init_polytope_4d(
    i_ga, i_gb, i_b, gjk_state: array_class.GJKState, collider_info: array_class.ColliderInfo
):
    """
    Create the polytope for EPA from a 3-simplex (tetrahedron).

    Returns
    -------
    int
        0 when successful, or a flag indicating an error.
    """
    flag = EPA_POLY_INIT_RETURN_CODE.SUCCESS

    # Insert simplex vertices into the polytope
    vi = qd.Vector([0, 0, 0, 0], dt=qd.i32)
    for i in range(4):
        vi[i] = func_epa_insert_vertex_to_polytope(
            i_b,
            gjk_state.simplex_vertex.id1[i_b, i],
            gjk_state.simplex_vertex.id2[i_b, i],
            gjk_state.simplex_vertex.obj1[i_b, i],
            gjk_state.simplex_vertex.obj2[i_b, i],
            gjk_state.simplex_vertex.local_obj1[i_b, i],
            gjk_state.simplex_vertex.local_obj2[i_b, i],
            gjk_state.simplex_vertex.mink[i_b, i],
            gjk_state,
        )

    # If origin is on any face of the tetrahedron, replace the simplex with a 2-simplex (triangle)
    dist2_min = collider_info.gjk.FLOAT_MAX[None]
    v1_min, v2_min, v3_min = vi[0], vi[1], vi[2]
    for i in range(4):
        # Vertex indices for the faces in the hexahedron
        v1, v2, v3 = vi[0], vi[1], vi[2]
        # Adjacent face indices for the faces in the hexahedron
        a1, a2, a3 = 1, 3, 2
        if i == 1:
            v1, v2, v3 = vi[0], vi[3], vi[1]
            a1, a2, a3 = 2, 3, 0
        elif i == 2:
            v1, v2, v3 = vi[0], vi[2], vi[3]
            a1, a2, a3 = 0, 3, 1
        elif i == 3:
            v1, v2, v3 = vi[3], vi[2], vi[1]
            a1, a2, a3 = 2, 0, 1

        dist2 = func_attach_face_to_polytope(i_b, v1, v2, v3, a1, a2, a3, gjk_state, collider_info)
        if dist2 < dist2_min:
            dist2_min = dist2
            v1_min, v2_min, v3_min = v1, v2, v3

        if dist2 < collider_info.gjk.FLOAT_MIN_SQ[None]:
            func_replace_simplex_3(i_b, v1, v2, v3, gjk_state)
            flag = EPA_POLY_INIT_RETURN_CODE.P4_FALLBACK3
            break

    if flag == EPA_POLY_INIT_RETURN_CODE.SUCCESS:
        if (
            func_origin_tetra_intersection(
                gjk_state.polytope_verts.mink[i_b, vi[0]],
                gjk_state.polytope_verts.mink[i_b, vi[1]],
                gjk_state.polytope_verts.mink[i_b, vi[2]],
                gjk_state.polytope_verts.mink[i_b, vi[3]],
            )
            == RETURN_CODE.FAIL
        ):
            # A flat contact terminates GJK with a nearly degenerate tetrahedron, on which this strict test flips
            # with rounding, differing across backends and kernels at single precision. Aborting would silently drop
            # a genuine contact, so rebuild from the face nearest the origin through the 2-simplex initialization,
            # which grants deep contacts its own leniency and still rejects a genuinely invalid state. The polytope
            # built above is reset alongside (see func_replace_simplex_3), so that rebuild starts from empty.
            func_replace_simplex_3(i_b, v1_min, v2_min, v3_min, gjk_state)
            flag = EPA_POLY_INIT_RETURN_CODE.P4_FALLBACK3

    if flag == EPA_POLY_INIT_RETURN_CODE.SUCCESS:
        # Initialize face map
        for i in qd.static(range(4)):
            gjk_state.polytope_faces_map[i_b, i] = i
            gjk_state.polytope_faces.map_idx[i_b, i] = i
        gjk_state.polytope.nfaces_map[i_b] = 4

    return flag


@qd.func
def func_epa_support(
    i_ga,
    i_gb,
    i_b,
    pos_a: qd.types.vector(3),
    quat_a: qd.types.vector(4),
    pos_b: qd.types.vector(3),
    quat_b: qd.types.vector(4),
    dir,
    dir_norm,
    collider_state: array_class.ColliderState,
    gjk_state: array_class.GJKState,
    dyn_info: array_class.DynInfo,
    collider_info: array_class.ColliderInfo,
    rigid_config: qd.template(),
    collider_static_config: qd.template(),
):
    """
    Find support points on the two objects using [dir] and insert them into the polytope.

    Parameters
    ----------
    dir: gs.qd_vec3
        Vector from [ga] (obj1) to [gb] (obj2).
    """
    d = gs.qd_vec3(1, 0, 0)
    if dir_norm > collider_info.gjk.FLOAT_MIN[None]:
        d = dir / dir_norm

    (
        support_point_obj1,
        support_point_obj2,
        support_point_localpos1,
        support_point_localpos2,
        support_point_id_obj1,
        support_point_id_obj2,
        support_point_minkowski,
    ) = func_support(
        i_ga,
        i_gb,
        i_b,
        d,
        pos_a,
        quat_a,
        pos_b,
        quat_b,
        shrink_sphere=False,
        collider_state=collider_state,
        gjk_state=gjk_state,
        dyn_info=dyn_info,
        collider_info=collider_info,
        rigid_config=rigid_config,
        collider_static_config=collider_static_config,
    )

    # Insert the support points into the polytope
    v_index = func_epa_insert_vertex_to_polytope(
        i_b,
        support_point_id_obj1,
        support_point_id_obj2,
        support_point_obj1,
        support_point_obj2,
        support_point_localpos1,
        support_point_localpos2,
        support_point_minkowski,
        gjk_state,
    )

    return v_index


@qd.func
def func_attach_face_to_polytope(
    i_b, i_v1, i_v2, i_v3, i_a1, i_a2, i_a3, gjk_state: array_class.GJKState, collider_info: array_class.ColliderInfo
):
    """
    Attach a face to the polytope.

    [i_v1, i_v2, i_v3] are the vertices of the face, [i_a1, i_a2, i_a3] are the adjacent faces.

    Returns
    -------
    float
        Squared distance of the face to the origin.
    """
    dist2 = 0.0

    n = gjk_state.polytope.nfaces[i_b]
    gjk_state.polytope_faces.verts_idx[i_b, n][0] = i_v1
    gjk_state.polytope_faces.verts_idx[i_b, n][1] = i_v2
    gjk_state.polytope_faces.verts_idx[i_b, n][2] = i_v3
    gjk_state.polytope_faces.adj_idx[i_b, n][0] = i_a1
    gjk_state.polytope_faces.adj_idx[i_b, n][1] = i_a2
    gjk_state.polytope_faces.adj_idx[i_b, n][2] = i_a3
    gjk_state.polytope.nfaces[i_b] += 1

    # Compute the squared distance of the face to the origin
    gjk_state.polytope_faces.normal[i_b, n], ret = func_project_origin_to_plane(
        gjk_state.polytope_verts.mink[i_b, i_v3],
        gjk_state.polytope_verts.mink[i_b, i_v2],
        gjk_state.polytope_verts.mink[i_b, i_v1],
        collider_info,
    )
    if ret == RETURN_CODE.SUCCESS:
        normal = gjk_state.polytope_faces.normal[i_b, n]
        gjk_state.polytope_faces.dist2[i_b, n] = normal.dot(normal)
        gjk_state.polytope_faces.map_idx[i_b, n] = -1  # No map index yet
        dist2 = gjk_state.polytope_faces.dist2[i_b, n]

    return dist2


@qd.func
def func_replace_simplex_3(i_b, i_v1, i_v2, i_v3, gjk_state: array_class.GJKState):
    """
    Replace the simplex with a 2-simplex (triangle) from polytope vertices, and reset the polytope so that a
    subsequent polytope initialization rebuilds it from empty.

    Parameters
    ----------
    i_v1, i_v2, i_v3: int
        Indices of the vertices in the polytope that will be used to form the triangle.
    """
    gjk_state.simplex.nverts[i_b] = 3
    for i in qd.static(range(3)):
        i_v = i_v1
        if i == 1:
            i_v = i_v2
        elif i == 2:
            i_v = i_v3
        gjk_state.simplex_vertex.obj1[i_b, i] = gjk_state.polytope_verts.obj1[i_b, i_v]
        gjk_state.simplex_vertex.obj2[i_b, i] = gjk_state.polytope_verts.obj2[i_b, i_v]
        gjk_state.simplex_vertex.local_obj1[i_b, i] = gjk_state.polytope_verts.local_obj1[i_b, i_v]
        gjk_state.simplex_vertex.local_obj2[i_b, i] = gjk_state.polytope_verts.local_obj2[i_b, i_v]
        gjk_state.simplex_vertex.id1[i_b, i] = gjk_state.polytope_verts.id1[i_b, i_v]
        gjk_state.simplex_vertex.id2[i_b, i] = gjk_state.polytope_verts.id2[i_b, i_v]
        gjk_state.simplex_vertex.mink[i_b, i] = gjk_state.polytope_verts.mink[i_b, i_v]

    # Reset polytope
    gjk_state.polytope.nverts[i_b] = 0
    gjk_state.polytope.nfaces[i_b] = 0
    gjk_state.polytope.nfaces_map[i_b] = 0


@qd.func
def func_safe_epa(
    i_ga,
    i_gb,
    i_b,
    pos_a: qd.types.vector(3),
    quat_a: qd.types.vector(4),
    pos_b: qd.types.vector(3),
    quat_b: qd.types.vector(4),
    collider_state: array_class.ColliderState,
    gjk_state: array_class.GJKState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    collider_info: array_class.ColliderInfo,
    rigid_config: qd.template(),
    collider_static_config: qd.template(),
):
    """
    Safe EPA algorithm to find the exact penetration depth and contact normal using the simplex constructed by GJK.

    This implementation is more robust than the one based on MuJoCo's implementation for the following reasons:
    1) It guarantees that the lower bound of the depth is always smaller than the upper bound, within the tolerance.
    2) This is because we acknowledge that polytope face normal could be unstable when the face is degenerate. Even
    in that case, we can robustly estimate the lower bound of the depth, which gives us more robust results.
    3) In determining the normal direction of a polytope face, we use origin and the polytope vertices altogether
    to get a more stable normal direction, rather than just the origin.
    """
    upper = collider_info.gjk.FLOAT_MAX[None]
    upper2 = collider_info.gjk.FLOAT_MAX_SQ[None]
    lower = gs.qd_float(0.0)
    tolerance = collider_info.gjk.tolerance[None]
    EPS = rigid_info.EPS[None]

    # Index of the nearest face
    nearest_i_f = gs.qd_int(-1)
    prev_nearest_i_f = gs.qd_int(-1)

    discrete = func_is_discrete_geoms(i_ga, i_gb, dyn_info)
    if discrete:
        # If the objects are discrete, we do not use tolerance.
        tolerance = rigid_info.EPS[None]

    k_max = collider_info.gjk.epa_max_iterations[None]
    for k in range(k_max):
        prev_nearest_i_f = nearest_i_f

        # Find the polytope face with the smallest distance to the origin
        lower2 = collider_info.gjk.FLOAT_MAX_SQ[None]

        for i in range(gjk_state.polytope.nfaces_map[i_b]):
            i_f = gjk_state.polytope_faces_map[i_b, i]
            face_dist2 = gjk_state.polytope_faces.dist2[i_b, i_f]

            if face_dist2 < lower2:
                lower2 = face_dist2
                nearest_i_f = i_f

        if lower2 > upper2 or nearest_i_f == -1:
            # Invalid face found, stop the algorithm (lower bound of depth is larger than upper bound)
            nearest_i_f = prev_nearest_i_f
            break

        # Find a new support point w from the nearest face's normal
        lower = qd.sqrt(lower2)
        dir = gjk_state.polytope_faces.normal[i_b, nearest_i_f]
        wi = func_epa_support(
            i_ga,
            i_gb,
            i_b,
            pos_a,
            quat_a,
            pos_b,
            quat_b,
            dir,
            1.0,
            collider_state,
            gjk_state,
            dyn_info,
            collider_info,
            rigid_config,
            collider_static_config,
        )
        w = gjk_state.polytope_verts.mink[i_b, wi]

        # The upper bound of depth at k-th iteration
        upper_k = w.dot(dir)
        if upper_k < upper:
            upper = upper_k
            upper2 = upper**2

        # If the upper bound and lower bound are close enough, we can stop the algorithm
        if (upper - lower) < tolerance:
            break

        if discrete:
            repeated = False
            for i in range(gjk_state.polytope.nverts[i_b]):
                if i == wi:
                    continue
                elif (
                    gjk_state.polytope_verts.id1[i_b, i] == gjk_state.polytope_verts.id1[i_b, wi]
                    and gjk_state.polytope_verts.id2[i_b, i] == gjk_state.polytope_verts.id2[i_b, wi]
                ):
                    # The vertex w is already in the polytope, so we do not need to add it again.
                    repeated = True
                    break
            if repeated:
                break

        gjk_state.polytope.horizon_w[i_b] = w

        # Compute horizon
        horizon_flag = func_epa_horizon(i_b, nearest_i_f, gjk_state, collider_info)

        if horizon_flag:
            # There was an error in the horizon construction, so the horizon edge is not a closed loop.
            nearest_i_f = -1
            break

        if gjk_state.polytope.horizon_nedges[i_b] < 3:
            # Should not happen, because at least three edges should be in the horizon from one deleted face.
            nearest_i_f = -1
            break

        # Check if the memory space is enough for attaching new faces
        nfaces = gjk_state.polytope.nfaces[i_b]
        nedges = gjk_state.polytope.horizon_nedges[i_b]
        if nfaces + nedges >= collider_info.gjk.polytope_max_faces[None]:
            # If the polytope is full, we cannot insert new faces
            break

        # Attach the new faces
        # print("Attaching new faces to the polytope")
        attach_flag = RETURN_CODE.SUCCESS
        for i in range(nedges):
            # Face id of the current face to attach
            i_f0 = nfaces + i
            # Face id of the next face to attach
            i_f1 = nfaces + (i + 1) % nedges

            horizon_i_f = gjk_state.polytope_horizon_data.face_idx[i_b, i]
            horizon_i_e = gjk_state.polytope_horizon_data.edge_idx[i_b, i]

            horizon_v1 = gjk_state.polytope_faces.verts_idx[i_b, horizon_i_f][horizon_i_e]
            horizon_v2 = gjk_state.polytope_faces.verts_idx[i_b, horizon_i_f][(horizon_i_e + 1) % 3]

            # Change the adjacent face index of the existing face
            gjk_state.polytope_faces.adj_idx[i_b, horizon_i_f][horizon_i_e] = i_f0

            # Attach the new face.
            # If this if the first face, will be adjacent to the face that will be attached last.
            adj_i_f_0 = i_f0 - 1 if (i > 0) else nfaces + nedges - 1
            adj_i_f_1 = horizon_i_f
            adj_i_f_2 = i_f1

            attach_flag = func_safe_attach_face_to_polytope(
                i_b,
                wi,
                horizon_v2,
                horizon_v1,
                adj_i_f_2,  # Previous face id
                adj_i_f_1,
                adj_i_f_0,  # Next face id
                gjk_state,
                collider_info,
            )
            if attach_flag != RETURN_CODE.SUCCESS:
                # Unrecoverable numerical issue
                break

            dist2 = gjk_state.polytope_faces.dist2[i_b, gjk_state.polytope.nfaces[i_b] - 1]
            if (dist2 >= lower2 - EPS) and (dist2 <= upper2 + EPS):
                # Store face in the map
                nfaces_map = gjk_state.polytope.nfaces_map[i_b]
                gjk_state.polytope_faces_map[i_b, nfaces_map] = i_f0
                gjk_state.polytope_faces.map_idx[i_b, i_f0] = nfaces_map
                gjk_state.polytope.nfaces_map[i_b] += 1

        if attach_flag != RETURN_CODE.SUCCESS:
            nearest_i_f = -1
            break

        # Clear the horizon data for the next iteration
        gjk_state.polytope.horizon_nedges[i_b] = 0

        if (gjk_state.polytope.nfaces_map[i_b] == 0) or (nearest_i_f == -1):
            # No face candidate left
            nearest_i_f = -1
            break

    if nearest_i_f != -1:
        # Nearest face found
        dist2 = gjk_state.polytope_faces.dist2[i_b, nearest_i_f]
        flag = func_safe_epa_witness(i_ga, i_gb, i_b, nearest_i_f, gjk_state, collider_info)
        if flag == RETURN_CODE.SUCCESS:
            gjk_state.n_witness[i_b] = 1
            gjk_state.distance[i_b] = -qd.sqrt(dist2)
        else:
            # Failed to compute witness points, so the objects are not colliding
            gjk_state.n_witness[i_b] = 0
            gjk_state.distance[i_b] = 0.0
            nearest_i_f = -1
    else:
        # No face found, so the objects are not colliding
        gjk_state.n_witness[i_b] = 0
        gjk_state.distance[i_b] = 0.0

    return nearest_i_f


@qd.func
def func_safe_epa_witness(
    i_ga, i_gb, i_b, i_f, gjk_state: array_class.GJKState, collider_info: array_class.ColliderInfo
):
    """
    Compute the witness points from the geometries for the face i_f of the polytope.
    """
    flag = RETURN_CODE.SUCCESS

    # Find the affine coordinates of the origin's projection on the face i_f
    face_iv1 = gjk_state.polytope_faces.verts_idx[i_b, i_f][0]
    face_iv2 = gjk_state.polytope_faces.verts_idx[i_b, i_f][1]
    face_iv3 = gjk_state.polytope_faces.verts_idx[i_b, i_f][2]
    face_v1 = gjk_state.polytope_verts.mink[i_b, face_iv1]
    face_v2 = gjk_state.polytope_verts.mink[i_b, face_iv2]
    face_v3 = gjk_state.polytope_verts.mink[i_b, face_iv3]

    # Project origin onto the face plane to get the barycentric coordinates
    proj_o, _ = func_project_origin_to_plane(face_v1, face_v2, face_v3, collider_info)
    _lambda = func_triangle_affine_coords(proj_o, face_v1, face_v2, face_v3)

    # Check validity of affine coordinates through reprojection
    v1 = gjk_state.polytope_verts.mink[i_b, face_iv1]
    v2 = gjk_state.polytope_verts.mink[i_b, face_iv2]
    v3 = gjk_state.polytope_verts.mink[i_b, face_iv3]

    proj_o_lambda = v1 * _lambda[0] + v2 * _lambda[1] + v3 * _lambda[2]
    reprojection_error = (proj_o - proj_o_lambda).norm()

    # Take into account the face magnitude, as the error is relative to the face size.
    max_edge_len_inv = qd.rsqrt(
        max((v1 - v2).norm_sqr(), (v2 - v3).norm_sqr(), (v3 - v1).norm_sqr(), collider_info.gjk.FLOAT_MIN_SQ[None])
    )
    rel_reprojection_error = reprojection_error * max_edge_len_inv
    # For smooth geometries (e.g. spheres), polytope faces become extremely small near convergence,
    # which amplifies the relative reprojection error even when the absolute error is negligible.
    # To avoid false rejections, we check both relative and absolute reprojection errors.
    if (
        rel_reprojection_error > collider_info.gjk.polytope_max_rel_reprojection_error[None]
        and reprojection_error > collider_info.gjk.polytope_max_abs_reprojection_error[None]
    ):
        flag = RETURN_CODE.FAIL

    if flag == RETURN_CODE.SUCCESS:
        # Point on geom 1
        v1 = gjk_state.polytope_verts.obj1[i_b, face_iv1]
        v2 = gjk_state.polytope_verts.obj1[i_b, face_iv2]
        v3 = gjk_state.polytope_verts.obj1[i_b, face_iv3]
        witness1 = v1 * _lambda[0] + v2 * _lambda[1] + v3 * _lambda[2]

        # Point on geom 2
        v1 = gjk_state.polytope_verts.obj2[i_b, face_iv1]
        v2 = gjk_state.polytope_verts.obj2[i_b, face_iv2]
        v3 = gjk_state.polytope_verts.obj2[i_b, face_iv3]
        witness2 = v1 * _lambda[0] + v2 * _lambda[1] + v3 * _lambda[2]

        gjk_state.witness.point_obj1[i_b, 0] = witness1
        gjk_state.witness.point_obj2[i_b, 0] = witness2

    return flag


@qd.func
def func_safe_epa_init(i_ga, i_gb, i_b, gjk_state: array_class.GJKState, collider_info: array_class.ColliderInfo):
    """
    Create the polytope for safe EPA from a 3-simplex (tetrahedron).

    Assume the tetrahedron is a non-degenerate simplex.
    """

    # Insert simplex vertices into the polytope
    vi = qd.Vector([0, 0, 0, 0], dt=qd.i32)
    for i in range(4):
        vi[i] = func_epa_insert_vertex_to_polytope(
            i_b,
            gjk_state.simplex_vertex.id1[i_b, i],
            gjk_state.simplex_vertex.id2[i_b, i],
            gjk_state.simplex_vertex.obj1[i_b, i],
            gjk_state.simplex_vertex.obj2[i_b, i],
            gjk_state.simplex_vertex.local_obj1[i_b, i],
            gjk_state.simplex_vertex.local_obj2[i_b, i],
            gjk_state.simplex_vertex.mink[i_b, i],
            gjk_state,
        )

    for i in range(4):
        # Vertex indices for the faces in the hexahedron
        v1, v2, v3 = vi[0], vi[1], vi[2]
        # Adjacent face indices for the faces in the hexahedron
        a1, a2, a3 = 1, 3, 2
        if i == 1:
            v1, v2, v3 = vi[0], vi[3], vi[1]
            a1, a2, a3 = 2, 3, 0
        elif i == 2:
            v1, v2, v3 = vi[0], vi[2], vi[3]
            a1, a2, a3 = 0, 3, 1
        elif i == 3:
            v1, v2, v3 = vi[3], vi[2], vi[1]
            a1, a2, a3 = 2, 0, 1

        func_safe_attach_face_to_polytope(i_b, v1, v2, v3, a1, a2, a3, gjk_state, collider_info)

    # Initialize face map
    for i in qd.static(range(4)):
        gjk_state.polytope_faces_map[i_b, i] = i
        gjk_state.polytope_faces.map_idx[i_b, i] = i
    gjk_state.polytope.nfaces_map[i_b] = 4


@qd.func
def func_safe_attach_face_to_polytope(
    i_b, i_v1, i_v2, i_v3, i_a1, i_a2, i_a3, gjk_state: array_class.GJKState, collider_info: array_class.ColliderInfo
):
    """
    Attach a face to the polytope.

    While attaching the face, 1) determine its normal direction, and 2) estimate the lower bound of the penetration
    depth in robust manner.

    [i_v1, i_v2, i_v3] are the vertices of the face, [i_a1, i_a2, i_a3] are the adjacent faces.
    """
    n = gjk_state.polytope.nfaces[i_b]
    gjk_state.polytope_faces.verts_idx[i_b, n][0] = i_v1
    gjk_state.polytope_faces.verts_idx[i_b, n][1] = i_v2
    gjk_state.polytope_faces.verts_idx[i_b, n][2] = i_v3
    gjk_state.polytope_faces.adj_idx[i_b, n][0] = i_a1
    gjk_state.polytope_faces.adj_idx[i_b, n][1] = i_a2
    gjk_state.polytope_faces.adj_idx[i_b, n][2] = i_a3
    gjk_state.polytope_faces.visited[i_b, n] = 0
    gjk_state.polytope.nfaces[i_b] += 1

    # Compute the normal of the plane
    normal, flag = func_plane_normal(
        gjk_state.polytope_verts.mink[i_b, i_v3],
        gjk_state.polytope_verts.mink[i_b, i_v2],
        gjk_state.polytope_verts.mink[i_b, i_v1],
        collider_info,
    )
    if flag == RETURN_CODE.SUCCESS:
        face_center = (
            gjk_state.polytope_verts.mink[i_b, i_v1]
            + gjk_state.polytope_verts.mink[i_b, i_v2]
            + gjk_state.polytope_verts.mink[i_b, i_v3]
        ) / 3.0

        # Use origin for initialization
        max_orient = -normal.dot(face_center)
        max_abs_orient = qd.abs(max_orient)

        # Consider other vertices in the polytope to reorient the normal
        nverts = gjk_state.polytope.nverts[i_b]
        for i_v in range(nverts):
            if i_v != i_v1 and i_v != i_v2 and i_v != i_v3:
                diff = gjk_state.polytope_verts.mink[i_b, i_v] - face_center
                orient = normal.dot(diff)
                if qd.abs(orient) > max_abs_orient:
                    max_abs_orient = qd.abs(orient)
                    max_orient = orient

        if max_orient > 0.0:
            normal = -normal

        gjk_state.polytope_faces.normal[i_b, n] = normal

        # Compute the safe lower bound of the penetration depth. We can do this by taking the minimum dot product
        # between the face normal and the vertices of the polytope face. This is safer than selecting one of the
        # vertices, because the face normal could be unstable, which ends up in significantly different dot product
        # values for different vertices.
        min_dist2 = collider_info.gjk.FLOAT_MAX[None]
        for i in qd.static(range(3)):
            i_v = i_v1
            if i == 1:
                i_v = i_v2
            elif i == 2:
                i_v = i_v3
            v = gjk_state.polytope_verts.mink[i_b, i_v]
            dist2 = normal.dot(v) ** 2
            if dist2 < min_dist2:
                min_dist2 = dist2
        dist2 = min_dist2
        gjk_state.polytope_faces.dist2[i_b, n] = dist2
        gjk_state.polytope_faces.map_idx[i_b, n] = -1  # No map index yet

    return flag


@qd.func
def func_plane_normal(v1, v2, v3, collider_info: array_class.ColliderInfo):
    """
    Compute the reliable normal of the plane defined by three points.
    """
    normal, flag = gs.qd_vec3(0.0, 0.0, 0.0), RETURN_CODE.FAIL
    finished = False

    d21 = v2 - v1
    d31 = v3 - v1
    d32 = v3 - v2

    # The three cross products are one quantity up to sign, so they differ only in rounding, and a sliver triangle
    # cancels catastrophically in some of the orderings while resolving in the others. Every ordering is therefore
    # tried before reporting the triangle degenerate: the whole point of the sequence is that a vanishing cross
    # product says nothing about the ones not yet computed. Reporting it instead discards the contact entirely.
    for i in qd.static(range(3)):
        if not finished:
            n = gs.qd_vec3(0.0, 0.0, 0.0)
            if i == 0:
                # Normal = (v1 - v2) x (v3 - v2)
                n = d32.cross(d21)
            elif i == 1:
                # Normal = (v2 - v1) x (v3 - v1)
                n = d21.cross(d31)
            else:
                # Normal = (v1 - v3) x (v2 - v3)
                n = d31.cross(d32)
            if n.norm() > collider_info.gjk.FLOAT_MIN[None]:
                normal = n.normalized()
                flag = RETURN_CODE.SUCCESS
                finished = True

    return normal, flag


from genesis.utils.deprecated_module_wrapper import create_virtual_deprecated_module

create_virtual_deprecated_module(__name__, "genesis.engine.solvers.rigid.gjk_decomp")
