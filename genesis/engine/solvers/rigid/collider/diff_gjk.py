import quadrants as qd
import genesis as gs
import genesis.utils.geom as gu
import genesis.utils.array_class as array_class
from . import gjk as GJK, contact, epa


@qd.func
def func_gjk_contact(
    i_ga,
    i_gb,
    i_b,
    ga_pos: qd.types.vector(3),
    ga_quat: qd.types.vector(4),
    gb_pos: qd.types.vector(3),
    gb_quat: qd.types.vector(4),
    pos_tol,
    normal_tol,
    geoms_init_AABB: array_class.GeomsInitAABB,
    dyn_state: array_class.DynState,
    collider_state: array_class.ColliderState,
    gjk_state: array_class.GJKState,
    diff_contact_input: array_class.DiffContactInput,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    collider_info: array_class.ColliderInfo,
    rigid_config: qd.template(),
    collider_static_config: qd.template(),
):
    """
    Detect multiple possible contact points between two geometries using GJK and EPA algorithms, and compute weights
    of the contact points for differentiability.

    We first run the GJK algorithm to find the minimum distance between the two geometries. If the distance is
    smaller than the collision epsilon, we consider the geometries colliding. If they are colliding, we run the extended
    EPA algorithm in the original configuration and the original EPA algorithm for multiple perturbed configurations.
    In this way, we can find multiple possible contact points.

    For each of the contact points, we first gather the non-differentiable contact data as follows:

    - Local position of the three vertices from the two geometries that form the face on the Minkowski difference.
    - Local position of the single vertex from the two geometries that form the support point for the face above.
    - Validity of the contact point (i.e., the face is not degenerate and the origin is not very close to the face). The
    contact point is valid only when the differentiable contact data can be computed in numerically stable way in both
    the forward and backward passes.

    Then, for each of the valid contact points, we compute the differentiable contact data as follows:

    - Contact position
    - Contact normal
    - Penetration depth
    - Weight of the contact point: It is multiplied by the penetration depth to get the differentiable penetration depth.

    Note that these terms can be computed from the non-differentiable contact data in a differentiable way. Therefore,
    we store the non-differentiable contact data along with the differentiable contact data for the backward pass.
    """
    EPS = rigid_info.EPS[None]

    # Clear the cache to prepare for this GJK-EPA run.
    GJK.clear_cache(i_b, gjk_state)

    gjk_state.n_diff_contact_input[i_b] = 0

    # Thread-local copies for perturbation (stored in registers, not global state)
    ga_pos_local = ga_pos
    ga_quat_local = ga_quat
    gb_pos_local = gb_pos
    gb_quat_local = gb_quat

    # Axis to rotate the geometry for perturbation
    axis_0 = qd.Vector.zero(gs.qd_float, 3)
    axis_1 = qd.Vector.zero(gs.qd_float, 3)

    # Default contact point and penetration in the original configuration
    default_contact_pos = gs.qd_vec3(0.0, 0.0, 0.0)
    default_penetration = gs.qd_float(0.0)
    found_default_epa = False

    # 4 (small) + 4 (large) perturbated configurations
    num_perturb = 8

    ### Detect multiple possible contact points and gather the non-differentiable contact data.
    for i in range(1 + num_perturb):
        # First iteration: Detect contact in default configuration.
        # 2,3,4,5: Detect contacts in slightly perturbed configuration.
        # 6,7,8,9: Detect contacts in more perturbed configuration.
        if i > 0:
            # Reset to the default configuration
            ga_pos_local = ga_pos
            ga_quat_local = ga_quat
            gb_pos_local = gb_pos
            gb_quat_local = gb_quat

            # Perturbation axis must not be aligned with the principal axes of inertia the geometry,
            # otherwise it would be more sensitive to ill-conditionning.
            axis = (2 * (i % 2) - 1) * axis_0 + (1 - 2 * ((i // 2) % 2)) * axis_1
            rotang = 1e-2 * (100 ** ((i - 1) // 4))
            qrot = gu.qd_rotvec_to_quat(rotang * axis, EPS)

            # Apply perturbation to local variables
            ga_pos_local, ga_quat_local = contact.func_rotate_frame(
                ga_pos_local, ga_quat_local, default_contact_pos, qrot
            )
            gb_pos_local, gb_quat_local = contact.func_rotate_frame(
                gb_pos_local, gb_quat_local, default_contact_pos, gu.qd_inv_quat(qrot)
            )

        gjk_flag = GJK.func_safe_gjk(
            i_ga,
            i_gb,
            i_b,
            ga_pos_local,
            ga_quat_local,
            gb_pos_local,
            gb_quat_local,
            collider_state,
            gjk_state,
            dyn_info,
            rigid_info,
            collider_info,
            rigid_config,
            collider_static_config,
        )

        if gjk_flag == GJK.GJK_RETURN_CODE.INTERSECT:
            # Initialize polytope
            gjk_state.polytope.nverts[i_b] = 0
            gjk_state.polytope.nfaces[i_b] = 0
            gjk_state.polytope.nfaces_map[i_b] = 0
            gjk_state.polytope.horizon_nedges[i_b] = 0

            # Construct the initial polytope from the GJK simplex
            epa.func_safe_epa_init(i_ga, i_gb, i_b, gjk_state, collider_info)

            if i == 0:
                # In default configuration, we use the extended EPA algorithm to find multiple contact points.
                max_epa_iter = collider_info.gjk.epa_max_iterations[None]
                while max_epa_iter > 0:
                    i_f, num_iter = func_extended_epa(
                        i_ga,
                        i_gb,
                        i_b,
                        ga_pos_local,
                        ga_quat_local,
                        gb_pos_local,
                        gb_quat_local,
                        max_epa_iter,
                        collider_state,
                        gjk_state,
                        dyn_info,
                        rigid_info,
                        collider_info,
                        rigid_config,
                        collider_static_config,
                    )
                    max_epa_iter -= num_iter

                    if i_f == -1:
                        break

                    # Mark the face as visited
                    gjk_state.polytope_faces.visited[i_b, i_f] = 1

                    # Compute penetration depth
                    witness1 = gjk_state.witness.point_obj1[i_b, 0]
                    witness2 = gjk_state.witness.point_obj2[i_b, 0]

                    normal = witness2 - witness1
                    penetration = normal.norm()

                    # If the penetration depth is larger than the (default EPA depth + eps), we can ignore this contact
                    # because the weight of the contact point would be 0.
                    if found_default_epa and (
                        penetration > default_penetration + collider_info.gjk.diff_contact_eps_distance[None]
                    ):
                        continue

                    # Add input data for differentiable contact detection
                    func_add_diff_contact_input(
                        i_ga,
                        i_gb,
                        i_b,
                        i_f,
                        ga_pos_local,
                        ga_quat_local,
                        gb_pos_local,
                        gb_quat_local,
                        collider_state,
                        gjk_state,
                        dyn_info,
                        collider_info,
                        rigid_config,
                        collider_static_config,
                    )

                    if not found_default_epa:
                        # If the default contact is already numerically unstable, we do not add any contact point.
                        if gjk_state.diff_contact_input.valid[i_b, 0] == 0:
                            gjk_state.n_diff_contact_input[i_b] = 0
                            break
                        default_contact_pos = 0.5 * (witness1 + witness2)
                        default_penetration = penetration

                        axis_0, axis_1 = func_contact_orthogonals(
                            i_ga, i_gb, i_b, normal / penetration, geoms_init_AABB, dyn_state, dyn_info, rigid_info
                        )

                        found_default_epa = True

                    # Break the loop if we found enough contact points for default configuration. As we can find at most
                    # 8 contact points for perturbed configurations, we can find at most max_contacts_per_pair - 8
                    # contact points for default configuration.
                    if gjk_state.n_diff_contact_input[i_b] >= (
                        collider_info.gjk.max_contacts_per_pair[None] - num_perturb
                    ):
                        break

                # If we failed to find the default contact point, we do not add any contact point.
                if not found_default_epa:
                    break
            else:
                i_f = epa.func_safe_epa(
                    i_ga,
                    i_gb,
                    i_b,
                    ga_pos_local,
                    ga_quat_local,
                    gb_pos_local,
                    gb_quat_local,
                    collider_state,
                    gjk_state,
                    dyn_info,
                    rigid_info,
                    collider_info,
                    rigid_config,
                    collider_static_config,
                )
                if i_f == -1:
                    continue

                # Add input data for differentiable contact detection
                func_add_diff_contact_input(
                    i_ga,
                    i_gb,
                    i_b,
                    i_f,
                    ga_pos,
                    ga_quat,
                    gb_pos,
                    gb_quat,
                    collider_state,
                    gjk_state,
                    dyn_info,
                    collider_info,
                    rigid_config,
                    collider_static_config,
                )

        elif i == 0:
            # If there was no intersection at the default configuration, we do not add any contact point.
            break

    ### Compute the differentiable contact data from the non-differentiable data.
    n_contacts = 0
    for i_c in range(gjk_state.n_diff_contact_input[i_b]):
        # We ignore the contact point if it is not numerically stable.
        if gjk_state.diff_contact_input.valid[i_b, i_c] == 0:
            continue

        # Compute the differentiable contact data.
        ref_penetration = -1.0
        if i_c > 0:
            ref_penetration = default_penetration
        contact_pos, contact_normal, penetration, weight = func_differentiable_contact(
            i_ga, i_gb, i_b, i_c, ref_penetration, dyn_state, diff_contact_input, collider_info
        )
        if i_c == 0:
            default_penetration = penetration

        # We ignore the contact point if the weight is 0.
        if weight == 0.0:
            if i_c == 0:
                # This will not happen, but we keep this for safety.
                break
            else:
                continue

        diff_penetration = penetration * weight

        # Check if there is any duplicate contact point.
        duplicate_id = -1
        for i_c2 in range(n_contacts):
            prev_contact_pos = gjk_state.contact_pos[i_b, i_c2]
            prev_contact_normal = gjk_state.normal[i_b, i_c2]

            if (contact_pos - prev_contact_pos).norm() > pos_tol:
                continue
            if contact_normal.dot(prev_contact_normal) < (1.0 - normal_tol):
                continue
            duplicate_id = i_c2
            break

        insert_id = n_contacts
        if duplicate_id != -1:
            # If it is duplicate and the prev. penetration depth is smaller, we replace the duplicate contact point.
            if gjk_state.diff_penetration[i_b, duplicate_id] < diff_penetration:
                insert_id = duplicate_id
            else:
                continue

        if i_c > 0 and insert_id == 0:
            # We keep the first contact point as the reference contact point.
            continue

        # Update the differentiable contact data.
        gjk_state.contact_pos[i_b, insert_id] = contact_pos
        gjk_state.normal[i_b, insert_id] = contact_normal
        gjk_state.diff_penetration[i_b, insert_id] = diff_penetration

        # Update the non-differentiable contact data for the backward pass.
        gjk_state.diff_contact_input.local_pos1_a[i_b, insert_id] = gjk_state.diff_contact_input.local_pos1_a[i_b, i_c]
        gjk_state.diff_contact_input.local_pos1_b[i_b, insert_id] = gjk_state.diff_contact_input.local_pos1_b[i_b, i_c]
        gjk_state.diff_contact_input.local_pos1_c[i_b, insert_id] = gjk_state.diff_contact_input.local_pos1_c[i_b, i_c]
        gjk_state.diff_contact_input.local_pos2_a[i_b, insert_id] = gjk_state.diff_contact_input.local_pos2_a[i_b, i_c]
        gjk_state.diff_contact_input.local_pos2_b[i_b, insert_id] = gjk_state.diff_contact_input.local_pos2_b[i_b, i_c]
        gjk_state.diff_contact_input.local_pos2_c[i_b, insert_id] = gjk_state.diff_contact_input.local_pos2_c[i_b, i_c]
        gjk_state.diff_contact_input.w_local_pos1[i_b, insert_id] = gjk_state.diff_contact_input.w_local_pos1[i_b, i_c]
        gjk_state.diff_contact_input.w_local_pos2[i_b, insert_id] = gjk_state.diff_contact_input.w_local_pos2[i_b, i_c]
        gjk_state.diff_contact_input.ref_id[i_b, insert_id] = 0
        if insert_id == 0:
            gjk_state.diff_contact_input.ref_penetration[i_b, insert_id] = penetration

        if insert_id == n_contacts:
            n_contacts += 1

        if n_contacts >= collider_info.gjk.max_contacts_per_pair[None]:
            break

    gjk_state.n_contacts[i_b] = n_contacts
    gjk_state.is_col[i_b] = n_contacts > 0
    gjk_state.multi_contact_flag[i_b] = True


@qd.func
def func_extended_epa(
    i_ga,
    i_gb,
    i_b,
    pos_a: qd.types.vector(3),
    quat_a: qd.types.vector(4),
    pos_b: qd.types.vector(3),
    quat_b: qd.types.vector(4),
    max_iter,
    collider_state: array_class.ColliderState,
    gjk_state: array_class.GJKState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    collider_info: array_class.ColliderInfo,
    rigid_config: qd.template(),
    collider_static_config: qd.template(),
):
    """
    Extended version of safe EPA algorithm to find multiple possible contact points for differentiable contact detection.

    While the original safe EPA algorithm finds the farthest boundary face on the Minkowski difference, this function
    can be used to find nearly-farthest boundary faces on the Minkowski difference. When the configurations of the
    objects are slightly perturbed, the farthest boundary face could change to one of the nearly-farthest boundary faces.
    Therefore, we use this function to find such nearly-farthest boundary faces for differentiability.
    """
    tolerance = collider_info.gjk.tolerance[None]
    nearest_i_f = gs.qd_int(-1)

    discrete = GJK.func_is_discrete_geoms(i_ga, i_gb, dyn_info)
    if discrete:
        # If the objects are discrete, we do not use tolerance.
        tolerance = rigid_info.EPS[None]

    k = 0
    converged = False
    while k < max_iter:
        k += 1

        # Find the polytope face with the smallest distance to the origin
        lower2 = collider_info.gjk.FLOAT_MAX_SQ[None]
        nearest_i_f = -1

        for i in range(gjk_state.polytope.nfaces_map[i_b]):
            i_f = gjk_state.polytope_faces_map[i_b, i]
            if gjk_state.polytope_faces.visited[i_b, i_f] == 1:
                continue

            face_dist2 = gjk_state.polytope_faces.dist2[i_b, i_f]
            if face_dist2 < lower2:
                lower2 = face_dist2
                nearest_i_f = i_f

        if nearest_i_f == -1:
            break

        # Find a new support point w from the nearest face's normal
        lower = qd.sqrt(lower2)
        dir = gjk_state.polytope_faces.normal[i_b, nearest_i_f]
        wi = epa.func_epa_support(
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
        upper = w.dot(dir)

        # If the upper bound and lower bound are close enough, we can stop the algorithm
        if (upper - lower) < tolerance:
            converged = True
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
                nearest_i_f = -1
                break

        gjk_state.polytope.horizon_w[i_b] = w

        # Compute horizon
        horizon_flag = epa.func_epa_horizon(i_b, nearest_i_f, gjk_state, collider_info)

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
            nearest_i_f = -1
            break

        # Attach the new faces
        attach_flag = GJK.RETURN_CODE.SUCCESS
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

            attach_flag = epa.func_safe_attach_face_to_polytope(
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
            if attach_flag != GJK.RETURN_CODE.SUCCESS:
                # Unrecoverable numerical issue
                break

            # Store face in the map
            nfaces_map = gjk_state.polytope.nfaces_map[i_b]
            gjk_state.polytope_faces_map[i_b, nfaces_map] = i_f0
            gjk_state.polytope_faces.map_idx[i_b, i_f0] = nfaces_map
            gjk_state.polytope.nfaces_map[i_b] += 1

        if attach_flag != GJK.RETURN_CODE.SUCCESS:
            nearest_i_f = -1
            break

        # Clear the horizon data for the next iteration
        gjk_state.polytope.horizon_nedges[i_b] = 0

        if (gjk_state.polytope.nfaces_map[i_b] == 0) or (nearest_i_f == -1):
            # No face candidate left
            nearest_i_f = -1
            break

    if converged:
        # Remove the last vertex from the polytope, because it was added because of this boundary face
        gjk_state.polytope.nverts[i_b] -= 1
    else:
        nearest_i_f = -1

    if nearest_i_f != -1:
        # Nearest face found
        dist2 = gjk_state.polytope_faces.dist2[i_b, nearest_i_f]
        flag = epa.func_safe_epa_witness(i_ga, i_gb, i_b, nearest_i_f, gjk_state, collider_info)
        if flag == GJK.RETURN_CODE.SUCCESS:
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

    return nearest_i_f, k


@qd.func
def func_add_diff_contact_input(
    i_ga,
    i_gb,
    i_b,
    i_f,
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
    Prepare the (non-differentiable) contact data that can be used for computing the differentiable contact data later.
    Therefore, this function is not differentiable.
    """
    n = gjk_state.n_diff_contact_input[i_b]

    i_v1 = gjk_state.polytope_faces.verts_idx[i_b, i_f][0]
    i_v2 = gjk_state.polytope_faces.verts_idx[i_b, i_f][1]
    i_v3 = gjk_state.polytope_faces.verts_idx[i_b, i_f][2]

    # Define the face (possibly) on the boundary of the Minkowski difference in the default configuration
    mink1 = gs.qd_vec3(0.0, 0.0, 0.0)
    mink2 = gs.qd_vec3(0.0, 0.0, 0.0)
    mink3 = gs.qd_vec3(0.0, 0.0, 0.0)

    for i in qd.static(range(3)):
        curr_i_v = i_v1
        if i == 1:
            curr_i_v = i_v2
        elif i == 2:
            curr_i_v = i_v3

        mink = func_compute_minkowski_point(
            pos_a,
            quat_a,
            pos_b,
            quat_b,
            gjk_state.polytope_verts.local_obj1[i_b, curr_i_v],
            gjk_state.polytope_verts.local_obj2[i_b, curr_i_v],
        )
        if i == 0:
            mink1 = mink
        elif i == 1:
            mink2 = mink
        elif i == 2:
            mink3 = mink

    ### Check validity of this contact. The contact is valid if the contact information could be computed in numerically
    ### stable way in both the forward and backward pass.
    # (a) Check if the face is degenerate.
    normal = func_plane_normal(mink1, mink2, mink3)
    normal_norm = normal.norm()
    is_face_degenerate = normal_norm < collider_info.gjk.diff_contact_min_normal_norm[None]

    # (b) Check if the origin is very close to the face (which means very small penetration depth).
    proj_o = func_project_origin_to_plane(mink1, mink2, mink3, normal)
    origin_dist = proj_o.norm()
    is_origin_close_to_face = origin_dist < collider_info.gjk.diff_contact_min_penetration[None]

    ### Orient the face normal, so that it points to the other side of the origin.
    face_center = (mink1 + mink2 + mink3) / 3.0
    if normal_norm > collider_info.gjk.FLOAT_MIN[None]:
        normal = normal.normalized()
    if normal.dot(face_center) < 0.0:
        normal = -normal

    ### Compute the support point along the face normal.
    obj1, obj2, localpos1, localpos2, id1, id2, mink = GJK.func_support(
        i_ga,
        i_gb,
        i_b,
        normal,
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

    gjk_state.diff_contact_input.local_pos1_a[i_b, n] = gjk_state.polytope_verts.local_obj1[i_b, i_v1]
    gjk_state.diff_contact_input.local_pos1_b[i_b, n] = gjk_state.polytope_verts.local_obj1[i_b, i_v2]
    gjk_state.diff_contact_input.local_pos1_c[i_b, n] = gjk_state.polytope_verts.local_obj1[i_b, i_v3]
    gjk_state.diff_contact_input.local_pos2_a[i_b, n] = gjk_state.polytope_verts.local_obj2[i_b, i_v1]
    gjk_state.diff_contact_input.local_pos2_b[i_b, n] = gjk_state.polytope_verts.local_obj2[i_b, i_v2]
    gjk_state.diff_contact_input.local_pos2_c[i_b, n] = gjk_state.polytope_verts.local_obj2[i_b, i_v3]
    gjk_state.diff_contact_input.w_local_pos1[i_b, n] = localpos1
    gjk_state.diff_contact_input.w_local_pos2[i_b, n] = localpos2
    gjk_state.diff_contact_input.valid[i_b, n] = not (is_face_degenerate or is_origin_close_to_face)
    gjk_state.n_diff_contact_input[i_b] += 1


@qd.func
def func_contact_orthogonals(
    i_ga,
    i_gb,
    i_b,
    normal: qd.types.vector(3),
    geoms_init_AABB: array_class.GeomsInitAABB,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
):
    EPS = rigid_info.EPS[None]

    axis_0 = qd.Vector.zero(gs.qd_float, 3)
    axis_1 = qd.Vector.zero(gs.qd_float, 3)

    # The reference geometry is the one that will have the largest impact on the position of
    # the contact point. Basically, the smallest one between the two, which can be approximated
    # by the volume of their respective bounding box.
    i_g = i_gb
    if dyn_info.geoms.type[i_ga] != gs.GEOM_TYPE.PLANE:
        size_ga = geoms_init_AABB[i_ga, 7]
        volume_ga = size_ga[0] * size_ga[1] * size_ga[2]
        size_gb = geoms_init_AABB[i_gb, 7]
        volume_gb = size_gb[0] * size_gb[1] * size_gb[2]
        i_g = i_ga if volume_ga < volume_gb else i_gb

    # Compute orthogonal basis mixing principal inertia axes of geometry with contact normal
    i_l = dyn_info.geoms.link_idx[i_g]
    rot = gu.qd_quat_to_R(dyn_state.links.i_quat[i_l, i_b], EPS)
    axis_idx = gs.qd_int(0)
    axis_angle_max = gs.qd_float(0.0)
    for i in qd.static(range(3)):
        axis_angle = qd.abs(rot[:, i].dot(normal))
        if axis_angle > axis_angle_max:
            axis_angle_max = axis_angle
            axis_idx = i
    axis_idx = (axis_idx + 1) % 3
    axis_0 = rot[:, axis_idx]
    axis_0 = (axis_0 - normal.dot(axis_0) * normal).normalized()
    axis_1 = normal.cross(axis_0)

    return axis_0, axis_1


@qd.func
def func_compute_minkowski_point(
    ga_pos: qd.types.vector(3),
    ga_quat: qd.types.vector(4),
    gb_pos: qd.types.vector(3),
    gb_quat: qd.types.vector(4),
    va: qd.types.vector(3),
    vb: qd.types.vector(3),
):
    # Transform the points to the global frame
    va_ = gu.qd_transform_by_trans_quat(va, ga_pos, ga_quat)
    vb_ = gu.qd_transform_by_trans_quat(vb, gb_pos, gb_quat)
    return va_ - vb_


# ------------------------------ Differentiable functions ------------------------------------
# These functions have minimal number of branches to align backward pass with forward pass.
# --------------------------------------------------------------------------------------------
@qd.func
def func_differentiable_contact(
    i_ga,
    i_gb,
    i_b,
    i_c,
    ref_penetration,
    dyn_state: array_class.DynState,
    diff_contact_input: array_class.DiffContactInput,
    collider_info: array_class.ColliderInfo,
):
    """
    Compute the contact normal, penetration, and point for contact [i_c] from the corresponding [diff_contact_input]
    in a differentiable way. The gradients flow through the position and quaternion stored in the [geoms_state].
    """
    eps_B = collider_info.gjk.diff_contact_eps_boundary[None]
    eps_D = collider_info.gjk.diff_contact_eps_distance[None]

    # Result
    contact_pos = gs.qd_vec3(0.0, 0.0, 0.0)
    contact_normal = gs.qd_vec3(0.0, 0.0, 0.0)
    penetration = gs.qd_float(0.0)
    weight = gs.qd_float(0.0)

    # Transformations of the geometries
    trans1 = dyn_state.geoms.pos[i_ga, i_b]
    trans2 = dyn_state.geoms.pos[i_gb, i_b]
    quat1 = dyn_state.geoms.quat[i_ga, i_b]
    quat2 = dyn_state.geoms.quat[i_gb, i_b]

    # Local positions of the vertices that form the contact
    local_pos1_a = diff_contact_input.local_pos1_a[i_b, i_c]
    local_pos1_b = diff_contact_input.local_pos1_b[i_b, i_c]
    local_pos1_c = diff_contact_input.local_pos1_c[i_b, i_c]
    local_pos2_a = diff_contact_input.local_pos2_a[i_b, i_c]
    local_pos2_b = diff_contact_input.local_pos2_b[i_b, i_c]
    local_pos2_c = diff_contact_input.local_pos2_c[i_b, i_c]

    # Support points of the contact
    w_local_pos1 = diff_contact_input.w_local_pos1[i_b, i_c]
    w_local_pos2 = diff_contact_input.w_local_pos2[i_b, i_c]

    # Compute global positions of the vertices
    pos1a = gu.qd_transform_by_trans_quat(local_pos1_a, trans1, quat1)
    pos1b = gu.qd_transform_by_trans_quat(local_pos1_b, trans1, quat1)
    pos1c = gu.qd_transform_by_trans_quat(local_pos1_c, trans1, quat1)
    pos2a = gu.qd_transform_by_trans_quat(local_pos2_a, trans2, quat2)
    pos2b = gu.qd_transform_by_trans_quat(local_pos2_b, trans2, quat2)
    pos2c = gu.qd_transform_by_trans_quat(local_pos2_c, trans2, quat2)

    # Compute the vertices on the Minkowski difference
    mink1 = pos1a - pos2a
    mink2 = pos1b - pos2b
    mink3 = pos1c - pos2c
    face_center = (mink1 + mink2 + mink3) / 3.0

    ### Compute the witness points on the two geometries.
    # Compute the normal of the face
    normal = func_plane_normal(mink1, mink2, mink3)

    # Project the origin onto the affine plane of the face: This operation is guaranteed to be numerically stable, as
    # the normal length is guaranteed to be larger than the minimum normal norm in [gjk_info].
    proj_o = func_project_origin_to_plane(mink1, mink2, mink3, normal)

    # Compute the affine coordinates of the origin's projection on the face: This operation is also guaranteed to be
    # numerically stable, as the normal length is guaranteed to be larger than the minimum normal norm in
    # [gjk_info].
    _lambda = func_triangle_affine_coords(mink1, mink2, mink3, normal, proj_o)

    # Point on geom 1
    w1 = pos1a * _lambda[0] + pos1b * _lambda[1] + pos1c * _lambda[2]
    # Point on geom 2
    w2 = pos2a * _lambda[0] + pos2b * _lambda[1] + pos2c * _lambda[2]

    ### Compute contact position, normal, and penetration depth. These operations are guaranteed to be numerically stable,
    ### especially the normalization of the contact normal, as the penetration depth is guaranteed to be larger than the
    ### minimum penetration depth in [gjk_info].
    contact_pos = 0.5 * (w1 + w2)
    contact_normal = (w2 - w1).normalized()
    penetration = (w2 - w1).norm()

    ### Compute weight of the contact point.
    face_normal = normal
    if normal.dot(face_center) < 0.0:
        face_normal = -normal
    face_normal = face_normal.normalized()

    w_pos1 = gu.qd_transform_by_trans_quat(w_local_pos1, trans1, quat1)
    w_pos2 = gu.qd_transform_by_trans_quat(w_local_pos2, trans2, quat2)
    w = w_pos1 - w_pos2

    # Boundary weight
    bsdist = qd.max(w.dot(face_normal) - face_center.dot(face_normal), 0.0)
    boundary_weight = 1.0 - qd.math.clamp(bsdist / eps_B, 0.0, 1.0)

    # Distance weight
    distance_weight = 1.0
    if ref_penetration >= 0.0:
        distance_weight = 1.0 - qd.math.clamp((penetration - ref_penetration) / eps_D, 0.0, 1.0)

    # Affine weight: Theoretically we need this, but in practice it could cause instability
    # FIXME: Can we stabilize it?
    # affine_weight_0 = 1.0 - qd.math.clamp(qd.max(0.0 - _lambda[0], _lambda[0] - 1.0) / eps_A, 0.0, 1.0)
    # affine_weight_1 = 1.0 - qd.math.clamp(qd.max(0.0 - _lambda[1], _lambda[1] - 1.0) / eps_A, 0.0, 1.0)
    # affine_weight_2 = 1.0 - qd.math.clamp(qd.max(0.0 - _lambda[2], _lambda[2] - 1.0) / eps_A, 0.0, 1.0)
    # affine_weight = (affine_weight_0 + affine_weight_1 + affine_weight_2) / 3.0
    affine_weight = 1.0

    # Compute final weight
    weight = affine_weight * distance_weight * boundary_weight

    return contact_pos, contact_normal, penetration, weight


@qd.func
def func_plane_contact_frame(
    i_b,
    i_ga,
    i_gb,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
):
    """World contact normal and convex-side radius of a plane [i_ga] vs convex [i_gb] pair: normal =
    -normalize(R(quat_plane) @ plane_local_dir), radius = data[0] for SPHERE / CAPSULE and 0 otherwise.

    Both the forward witness capture and the differentiable reconstruction share this frame (and the convention
    contact_pos = v - 0.5 * penetration * normal built on it)."""
    plane_dir = gs.qd_vec3(dyn_info.geoms.data[i_ga][0], dyn_info.geoms.data[i_ga][1], dyn_info.geoms.data[i_ga][2])
    plane_dir = gu.qd_transform_by_quat(plane_dir, dyn_state.geoms.quat[i_ga, i_b])
    normal = -plane_dir.normalized()

    radius = gs.qd_float(0.0)
    geom_type = dyn_info.geoms.type[i_gb]
    if geom_type == gs.GEOM_TYPE.SPHERE or geom_type == gs.GEOM_TYPE.CAPSULE:
        radius = dyn_info.geoms.data[i_gb][0]
    return normal, radius


@qd.func
def func_differentiable_sphere_contact(
    i_ga, i_gb, i_b, dyn_state: array_class.DynState, dyn_info: array_class.DynInfo, rigid_info: array_class.RigidInfo
):
    """Differentiable sphere-sphere contact reconstruction.

    The contact is a smooth closed form of the two geom centers, so gradients flow directly through
    [dyn_state.geoms.pos]; the Minkowski-triangle reconstruction of func_differentiable_contact is degenerate for two
    smoothly-curved surfaces, on which EPA never converges. Must match func_sphere_sphere_contact in
    capsule_contact.py.

    At coincident centers the distance is a non-differentiable cusp: the normal (delta / dist) and the distance's own
    gradient (also delta / dist) are both 0/0 there, which reverse-mode turns into NaN. The sqrt only runs when the
    centers are separated; otherwise the same arbitrary direction and constant penetration as the forward hold (the
    contact is degenerate there anyway).
    """
    EPS = rigid_info.EPS[None]

    radius_a = dyn_info.geoms.data[i_ga][0]
    radius_b = dyn_info.geoms.data[i_gb][0]
    delta = dyn_state.geoms.pos[i_ga, i_b] - dyn_state.geoms.pos[i_gb, i_b]
    dist_sq = delta.dot(delta)

    contact_normal = qd.Vector([1.0, 0.0, 0.0], dt=gs.qd_float)
    penetration = radius_a + radius_b
    if dist_sq > EPS * EPS:
        dist = qd.sqrt(dist_sq)
        contact_normal = delta / dist
        penetration = radius_a + radius_b - dist

    contact_pos = dyn_state.geoms.pos[i_ga, i_b] - (radius_a - 0.5 * penetration) * contact_normal
    return contact_pos, contact_normal, penetration, gs.qd_float(1.0)


@qd.func
def func_differentiable_plane_contact(
    i_ga,
    i_gb,
    i_b,
    i_c,
    dyn_state: array_class.DynState,
    diff_contact_input: array_class.DiffContactInput,
    dyn_info: array_class.DynInfo,
):
    """Differentiable plane-convex contact reconstruction.

    Mirrors the analytic plane branch of func_convex_convex_contact:
        normal      = -normalize(R(quat_plane) @ plane_local_dir)
        v_world     = R(quat_convex) @ core_local + pos_convex + radius * normal
        penetration = normal . (v_world - pos_plane)
        contact_pos = v_world - 0.5 * penetration * normal

    [i_ga] is the plane geom and [i_gb] the convex geom. [core_local] (box vertex / sphere center / capsule nearest
    endpoint, in the convex geom's local frame) is the pose-independent witness stored by
    kernel_fill_diff_contact_input_analytic; [radius] and the plane direction come from the geoms info. Gradients flow to
    both geom poses through the geoms state pos / quat. For a sphere, [core_local] is the local origin, so the
    orientation gradient is zero, matching the rotation-invariant forward contact.
    """
    trans_plane = dyn_state.geoms.pos[i_ga, i_b]
    trans_convex = dyn_state.geoms.pos[i_gb, i_b]
    quat_convex = dyn_state.geoms.quat[i_gb, i_b]

    normal, radius = func_plane_contact_frame(i_b, i_ga, i_gb, dyn_state, dyn_info)

    core_local = diff_contact_input.core_local[i_b, i_c]
    core_world = gu.qd_transform_by_trans_quat(core_local, trans_convex, quat_convex)
    v_world = core_world + radius * normal

    penetration = normal.dot(v_world - trans_plane)
    contact_pos = v_world - 0.5 * penetration * normal
    weight = gs.qd_float(1.0)
    return contact_pos, normal, penetration, weight


@qd.func
def func_plane_normal(v1, v2, v3):
    """
    Compute the normal of the plane defined by three points. The length of the normal corresponds to the two times the
    area of the triangle.
    """
    d21 = v2 - v1
    d31 = v3 - v1
    normal = d21.cross(d31)
    return normal


@qd.func
def func_project_origin_to_plane(v1, v2, v3, normal):
    """
    Project the origin onto the plane defined by a point on the plane and its normal.

    @ normal: The face normal computed as (v2 - v1) x (v3 - v1). Its length should be guaranteed to be larger than the
    minimum normal norm in [gjk_info], but we do not check it here.
    """
    # Since normal norm is guaranteed to be larger than sqrt(10 * EPS), [nn] is guaranteed to be larger than 10 * EPS.
    v = v1
    nv = normal.dot(v)
    nn = normal.norm_sqr()
    return normal * (nv / nn)


@qd.func
def func_triangle_affine_coords(v1, v2, v3, normal, point):
    """
    Compute the affine coordinates of the point with respect to the triangle.

    @ normal: The face normal computed as (v2 - v1) x (v3 - v1). Its length should be guaranteed to be larger than the
    minimum normal norm in [gjk_info], but we do not check it here.
    @ point: The point on the plane that we want to compute the affine coordinates of.
    """
    # Since normal norm is guaranteed to be larger than sqrt(10 * EPS), [nn] is guaranteed to be larger than 10 * EPS.
    nn = normal.norm_sqr()
    inv_nn = 1.0 / nn

    return gs.qd_vec3(
        (v2 - point).cross(v3 - point).dot(normal) * inv_nn,
        (v3 - point).cross(v1 - point).dot(normal) * inv_nn,
        (v1 - point).cross(v2 - point).dot(normal) * inv_nn,
    )


from genesis.utils.deprecated_module_wrapper import create_virtual_deprecated_module

create_virtual_deprecated_module(__name__, "genesis.engine.solvers.rigid.diff_gjk_decomp")
