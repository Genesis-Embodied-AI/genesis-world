import math

import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
import genesis.utils.array_class as array_class
from .constants import RETURN_CODE, GJK_RETURN_CODE, EPA_POLY_INIT_RETURN_CODE
from .gjk_utils import (
    func_ray_triangle_intersection,
    func_triangle_affine_coords,
    func_point_triangle_intersection,
    func_point_plane_same_side,
    func_origin_tetra_intersection,
    func_project_origin_to_plane,
)
from .utils import func_is_discrete_geoms, func_is_equal_vec, func_det3
from . import support_field

# Import support functions that are shared with epa
from .gjk_support import func_support, support_driver, support_mesh

# Import EPA functions directly
from .epa import (
    func_epa_init_polytope_2d,
    func_epa_init_polytope_3d,
    func_epa_init_polytope_4d,
    func_epa,
    func_safe_epa_init,
    func_safe_epa,
)

# Import multi_contact functions directly
from .multi_contact import func_safe_normalize, func_multi_contact


class GJK:
    def __init__(self, rigid_solver):
        self._solver = rigid_solver

        # Resolve the 'enable_contact_patch' option in place, so later consumers (the collider's convex contact
        # budget) read the resolved value. The contact patch (polygon clipping over the touching faces) skips
        # perturbation whenever it ran (see multi_contact_flag). It defaults on under MuJoCo compatibility with GJK
        # because the reference engine builds its manifolds that way inside its GJK/EPA and runs its perturbation
        # scheme only in its legacy fallback pipeline, which the MPR compatibility pipeline mirrors. It defaults off
        # otherwise, as the perturbation-based detection is more reliable on marginal manifolds.
        enable_contact_patch = rigid_solver._options.enable_contact_patch
        if enable_contact_patch is None:
            enable_contact_patch = (
                rigid_solver._enable_mujoco_compatibility
                and rigid_solver._options.use_gjk_collision
                and rigid_solver._enable_multi_contact
            )
        elif enable_contact_patch:
            if not rigid_solver._options.use_gjk_collision:
                gs.raise_exception("'enable_contact_patch' requires 'use_gjk_collision'.")
            if not rigid_solver._enable_multi_contact:
                gs.raise_exception("'enable_contact_patch' requires 'enable_multi_contact'.")
        rigid_solver._options.enable_contact_patch = enable_contact_patch
        gjk_max_iterations = 50
        epa_max_iterations = 50
        # 6 * epa_max_iterations is the maximum number of faces in the polytope.
        polytope_max_faces = 6 * epa_max_iterations

        if rigid_solver.rigid_config.requires_grad:
            # For differentiable contact detection, we find multiple contact points for each pair.
            max_contacts_per_pair = 20
            max_contact_polygon_verts = 1
        elif enable_contact_patch:
            # The maximum number of contacts per pair is related to the maximum number of contact manifold vertices.
            # MuJoCo sets [max_contacts_per_pair] to 50 and [max_contact_polygon_verts] to 150, when it uses
            # multi-contact detection algorithm, assuming that the faces could have more than 4 vertices. However, we
            # set them to smaller values, because we do not expect the faces to have more than 4 vertices in most cases,
            # and we want to keep the memory usage low.
            max_contacts_per_pair = 8
            max_contact_polygon_verts = 30
        else:
            max_contacts_per_pair = 1
            max_contact_polygon_verts = 1

        self.gjk_config = array_class.GJKStaticConfig(enable_contact_patch=enable_contact_patch)

        # Initialize GJK info
        self._gjk_info = array_class.get_gjk_info(
            max_contacts_per_pair=max_contacts_per_pair,
            max_contact_polygon_verts=max_contact_polygon_verts,
            gjk_max_iterations=gjk_max_iterations,
            epa_max_iterations=epa_max_iterations,
            # When using larger minimum values (e.g. gs.EPS), unstability could occur for some examples (e.g. box
            # pyramid). Also, since different backends could have different precisions (e.g. computing vector norm),
            # we use a very small value, so that there is no discrepancy between backends.
            FLOAT_MIN=1e-15,
            FLOAT_MAX=1e15,
            tolerance=1e-6,
            collision_eps=1e-6,
            # This value has been experimentally determined based on the examples that we currently have (e.g. pyramid,
            # tower, ...), but it could be further tuned based on the future examples.
            simplex_max_degeneracy_sq=1e-5**2,
            polytope_max_faces=polytope_max_faces,
            # This value has been experimentally determined based on the examples that we currently have (e.g. pyramid,
            # tower, ...). We observed the error usually reaches around 5e-4, so we set the threshold to 1e-4 to be
            # safe. However, this value could be further tuned based on the future examples.
            polytope_max_rel_reprojection_error=1e-4,
            polytope_max_abs_reprojection_error=1e-4,
            # The values are matching MuJoCo for compatibility. Increasing this value could be useful for detecting
            # contact manifolds even when the normals are not perfectly aligned, but we observed that it leads to more
            # false positives and thus not a perfect solution for the multi-contact detection.
            contact_face_tol=math.cos(1.6e-3),
            contact_edge_tol=math.sin(1.6e-3),
            # FIXME: Adjust these values based on the case study.
            diff_contact_eps_boundary=1e-2,
            diff_contact_eps_distance=1e-2,
            diff_contact_eps_affine=1e-2,
            # We apply sqrt to the 10 * EPS value because we often use the square of the normal norm as the denominator.
            diff_contact_min_normal_norm=math.sqrt(gs.EPS * 10.0),
            diff_contact_min_penetration=gs.EPS * 100.0,
        )

        # Initialize GJK state
        self.gjk_state = array_class.get_gjk_state(
            rigid_solver._B, rigid_solver.rigid_config, self._gjk_info, False, rigid_solver.rigid_config.requires_grad
        )

        self._is_active = False

    def activate(self):
        if self._is_active:
            return

        self.gjk_state = array_class.get_gjk_state(
            self._solver._B, self._solver.rigid_config, self._gjk_info, True, self._solver.rigid_config.requires_grad
        )
        self._is_active = True

    @property
    def is_active(self):
        return self._is_active


@qd.func
def func_compare_sign(a, b):
    """
    Compare the sign of two values.
    """
    ret = 0
    if a > 0 and b > 0:
        ret = 1
    elif a < 0 and b < 0:
        ret = -1
    return ret


@qd.func
def clear_cache(i_b, gjk_state: array_class.GJKState):
    """
    Clear the cache information to prepare for the next GJK-EPA run.

    The cache includes the temporary information about simplex consturction or multi-contact detection.
    """
    gjk_state.support_mesh_prev_vertex_id[i_b, 0] = -1
    gjk_state.support_mesh_prev_vertex_id[i_b, 1] = -1
    gjk_state.multi_contact_flag[i_b] = False
    gjk_state.last_searched_simplex_vertex_id[i_b] = 0


@qd.func
def func_gjk_contact(
    i_ga,
    i_gb,
    i_b,
    pos_a: qd.types.vector(3),
    quat_a: qd.types.vector(4),
    pos_b: qd.types.vector(3),
    quat_b: qd.types.vector(4),
    dyn_state: array_class.DynState,
    collider_state: array_class.ColliderState,
    gjk_state: array_class.GJKState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    collider_info: array_class.ColliderInfo,
    rigid_config: qd.template(),
    collider_static_config: qd.template(),
    gjk_static_config: qd.template(),
):
    """
    Detect (possibly multiple) contact between two geometries using GJK and EPA algorithms.

    We first run the GJK algorithm to find the minimum distance between the two geometries. If the distance is
    smaller than the collision epsilon, we consider the geometries colliding. If they are colliding, we run the EPA
    algorithm to find the exact contact points and normals.

    .. seealso::
    MuJoCo's implementation:
    https://github.com/google-deepmind/mujoco/blob/7dc7a349c5ba2db2d3f8ab50a367d08e2f1afbbc/src/engine/engine_collision_gjk.c#L2259
    """
    # Clear the cache to prepare for this GJK-EPA run
    clear_cache(i_b, gjk_state)

    # No EPA penetration face yet; set to the nearest face index below if EPA runs.
    gjk_state.nearest_face[i_b] = -1

    # We use MuJoCo's GJK implementation when the compatibility mode is enabled
    if qd.static(rigid_config.enable_mujoco_compatibility):
        # If any one of the geometries is a sphere or capsule, which are sphere-swept primitives,
        # we can shrink them to a point or line to detect shallow penetration faster
        is_sphere_swept_geom_a, is_sphere_swept_geom_b = (
            func_is_sphere_swept_geom(i_ga, dyn_info),
            func_is_sphere_swept_geom(i_gb, dyn_info),
        )
        shrink_sphere = is_sphere_swept_geom_a or is_sphere_swept_geom_b

        # Run GJK
        for _ in range(2 if shrink_sphere else 1):
            distance = func_gjk(
                i_ga,
                i_gb,
                i_b,
                pos_a,
                quat_a,
                pos_b,
                quat_b,
                shrink_sphere,
                collider_state,
                gjk_state,
                dyn_info,
                collider_info,
                rigid_config,
                collider_static_config,
            )

            if shrink_sphere:
                # If we shrunk the sphere and capsule to point and line and the distance is larger than the collision
                # epsilon, it means a shallow penetration. Thus we subtract the radius of the sphere and the capsule to
                # get the actual distance. If the distance is smaller than the collision epsilon, it means a deep
                # penetration, which requires the default GJK handling.
                if distance > collider_info.gjk.collision_eps[None]:
                    radius_a, radius_b = 0.0, 0.0
                    if is_sphere_swept_geom_a:
                        radius_a = dyn_info.geoms.data[i_ga][0]
                    if is_sphere_swept_geom_b:
                        radius_b = dyn_info.geoms.data[i_gb][0]

                    wa = gjk_state.witness.point_obj1[i_b, 0]
                    wb = gjk_state.witness.point_obj2[i_b, 0]
                    n = func_safe_normalize(wb - wa, collider_info)

                    gjk_state.distance[i_b] = distance - (radius_a + radius_b)
                    gjk_state.witness.point_obj1[i_b, 0] = wa + (radius_a * n)
                    gjk_state.witness.point_obj2[i_b, 0] = wb - (radius_b * n)

                    break

            # Only try shrinking the sphere once
            shrink_sphere = False

            distance = gjk_state.distance[i_b]
            nsimplex = gjk_state.nsimplex[i_b]
            collided = distance < collider_info.gjk.collision_eps[None]

            # To run EPA, we need following conditions:
            # 1. We did not find min. distance with shrink_sphere flag
            # 2. We have a valid GJK simplex (nsimplex > 0)
            # 3. We have a collision (distance < collision_epsilon)
            do_epa = (not shrink_sphere) and collided and (nsimplex > 0)

            if do_epa:
                # Initialize polytope
                gjk_state.polytope.nverts[i_b] = 0
                gjk_state.polytope.nfaces[i_b] = 0
                gjk_state.polytope.nfaces_map[i_b] = 0
                gjk_state.polytope.horizon_nedges[i_b] = 0

                # Construct the initial polytope from the GJK simplex
                polytope_flag = EPA_POLY_INIT_RETURN_CODE.SUCCESS
                if nsimplex == 2:
                    polytope_flag = func_epa_init_polytope_2d(
                        i_ga,
                        i_gb,
                        i_b,
                        pos_a,
                        quat_a,
                        pos_b,
                        quat_b,
                        collider_state,
                        gjk_state,
                        dyn_info,
                        rigid_info,
                        collider_info,
                        rigid_config,
                        collider_static_config,
                    )
                elif nsimplex == 4:
                    polytope_flag = func_epa_init_polytope_4d(i_ga, i_gb, i_b, gjk_state, collider_info)

                # Polytope 3D could be used as a fallback for 2D and 4D cases
                if (
                    nsimplex == 3
                    or (polytope_flag == EPA_POLY_INIT_RETURN_CODE.P2_FALLBACK3)
                    or (polytope_flag == EPA_POLY_INIT_RETURN_CODE.P4_FALLBACK3)
                ):
                    polytope_flag = func_epa_init_polytope_3d(
                        i_ga,
                        i_gb,
                        i_b,
                        pos_a,
                        quat_a,
                        pos_b,
                        quat_b,
                        collider_state,
                        gjk_state,
                        dyn_info,
                        collider_info,
                        rigid_config,
                        collider_static_config,
                    )

                # Assume touching. Zeroed only after the initialization above, whose 2-simplex path rejects a
                # separated near-miss by reading the distance GJK reported (see func_epa_init_polytope_3d).
                gjk_state.distance[i_b] = 0

                # Run EPA from the polytope
                if polytope_flag == EPA_POLY_INIT_RETURN_CODE.SUCCESS:
                    i_f = func_epa(
                        i_ga,
                        i_gb,
                        i_b,
                        pos_a,
                        quat_a,
                        pos_b,
                        quat_b,
                        collider_state,
                        gjk_state,
                        dyn_info,
                        collider_info,
                        rigid_config,
                        collider_static_config,
                    )
                    gjk_state.nearest_face[i_b] = i_f

                    if qd.static(gjk_static_config.enable_contact_patch):
                        # The patch needs a valid nearest polytope face and two discrete geometries to clip.
                        if i_f >= 0 and func_is_discrete_geoms(i_ga, i_gb, dyn_info):
                            func_multi_contact(
                                i_ga, i_gb, i_b, i_f, pos_a, quat_a, pos_b, quat_b, gjk_state, dyn_info, collider_info
                            )
                            gjk_state.multi_contact_flag[i_b] = True
    else:
        gjk_flag = func_safe_gjk(
            i_ga,
            i_gb,
            i_b,
            pos_a,
            quat_a,
            pos_b,
            quat_b,
            collider_state,
            gjk_state,
            dyn_info,
            rigid_info,
            collider_info,
            rigid_config,
            collider_static_config,
        )
        if gjk_flag == GJK_RETURN_CODE.INTERSECT:
            # Initialize polytope
            gjk_state.polytope.nverts[i_b] = 0
            gjk_state.polytope.nfaces[i_b] = 0
            gjk_state.polytope.nfaces_map[i_b] = 0
            gjk_state.polytope.horizon_nedges[i_b] = 0

            # Construct the initial polytope from the GJK simplex
            func_safe_epa_init(i_ga, i_gb, i_b, gjk_state, collider_info)

            # Run EPA from the polytope
            gjk_state.nearest_face[i_b] = func_safe_epa(
                i_ga,
                i_gb,
                i_b,
                pos_a,
                quat_a,
                pos_b,
                quat_b,
                collider_state,
                gjk_state,
                dyn_info,
                rigid_info,
                collider_info,
                rigid_config,
                collider_static_config,
            )

            if qd.static(gjk_static_config.enable_contact_patch):
                # The patch needs a valid nearest polytope face and two discrete geometries to clip.
                i_f = gjk_state.nearest_face[i_b]
                if i_f >= 0 and func_is_discrete_geoms(i_ga, i_gb, dyn_info):
                    func_multi_contact(
                        i_ga, i_gb, i_b, i_f, pos_a, quat_a, pos_b, quat_b, gjk_state, dyn_info, collider_info
                    )
                    gjk_state.multi_contact_flag[i_b] = True

    # Compute the final contact points and normals
    n_contacts = 0
    gjk_state.is_col[i_b] = gjk_state.distance[i_b] < 0.0
    gjk_state.penetration[i_b] = -gjk_state.distance[i_b] if gjk_state.is_col[i_b] else 0.0

    if gjk_state.is_col[i_b]:
        for i in range(gjk_state.n_witness[i_b]):
            w1 = gjk_state.witness.point_obj1[i_b, i]
            w2 = gjk_state.witness.point_obj2[i_b, i]
            contact_pos = 0.5 * (w1 + w2)

            normal = w2 - w1
            normal_len = normal.norm()
            if normal_len < collider_info.gjk.FLOAT_MIN[None]:
                continue

            normal = normal / normal_len

            # For a penetrating contact the contact normal is the EPA nearest-face normal (the Minkowski penetration
            # direction), which is well-conditioned. The witness-point difference w2 - w1 is the tiny penetration
            # vector built from barycentric blends of large (geom-scale) support points, so in fp32 it is numerically
            # fragile for a shallow contact and can tilt the normal by tens of degrees (catastrophic for grazing
            # convex-decomposition piece contacts). It is kept only as the sign reference, and as the normal itself
            # when no polytope was built (nearest_face < 0). Mujoco-compatibility mode keeps the witness normal to
            # reproduce MuJoCo's contact set exactly.
            if qd.static(not rigid_config.enable_mujoco_compatibility):
                i_f = gjk_state.nearest_face[i_b]
                if i_f >= 0:
                    face_normal = gjk_state.polytope_faces.normal[i_b, i_f]
                    if face_normal.dot(normal) < 0.0:
                        face_normal = -face_normal
                    normal = face_normal

            gjk_state.contact_pos[i_b, n_contacts] = contact_pos
            gjk_state.normal[i_b, n_contacts] = normal
            n_contacts += 1

    gjk_state.n_contacts[i_b] = n_contacts
    # If there are no contacts, we set the penetration and is_col to 0
    # FIXME: When we use if statement here, it leads to a bug in some backends (e.g. x86 cpu). Need to investigate.
    gjk_state.is_col[i_b] = False if n_contacts == 0 else gjk_state.is_col[i_b]
    gjk_state.penetration[i_b] = 0.0 if n_contacts == 0 else gjk_state.penetration[i_b]


@qd.func
def func_gjk(
    i_ga,
    i_gb,
    i_b,
    pos_a: qd.types.vector(3),
    quat_a: qd.types.vector(4),
    pos_b: qd.types.vector(3),
    quat_b: qd.types.vector(4),
    shrink_sphere,
    collider_state: array_class.ColliderState,
    gjk_state: array_class.GJKState,
    dyn_info: array_class.DynInfo,
    collider_info: array_class.ColliderInfo,
    rigid_config: qd.template(),
    collider_static_config: qd.template(),
):
    """
    GJK algorithm to compute the minimum distance between two convex objects.

    This implementation is based on the MuJoCo implementation.

    TODO: This implementation could be further improved by referencing the follow-up work shown below.

    Parameters
    ----------
    shrink_sphere: bool
        If True, use point and line support functions for sphere and capsule geometries, respectively. It is more
        efficient and stable for shallow penetrations than the full GJK algorithm. However, if there is a deep
        penetration, we have to fallback to the full GJK algorithm by setting this parameter to False.

    .. seealso::
    MuJoCo's original implementation:
    https://github.com/google-deepmind/mujoco/blob/7dc7a349c5ba2db2d3f8ab50a367d08e2f1afbbc/src/engine/engine_collision_gjk.c#L171

    Original paper:
    Gilbert, Elmer G., Daniel W. Johnson, and S. Sathiya Keerthi.
    "A fast procedure for computing the distance between complex objects in three-dimensional space."
    IEEE Journal on Robotics and Automation 4.2 (2002): 193-203.

    Further improvements:
    Cameron, Stephen. "Enhancing GJK: Computing minimum and penetration distances between convex polyhedra."
    Proceedings of international conference on robotics and automation. Vol. 4. IEEE, 1997.
    https://www.cs.ox.ac.uk/people/stephen.cameron/distances/gjk2.4/

    Montaut, Louis, et al. "Collision detection accelerated: An optimization perspective."
    https://arxiv.org/abs/2205.09663
    """
    # Simplex index
    n = gs.qd_int(0)
    # Final number of simplex vertices
    nsimplex = gs.qd_int(0)
    # Number of witness points and distance
    nx = gs.qd_int(0)
    dist = gs.qd_float(0.0)
    # Lambda for barycentric coordinates
    _lambda = gs.qd_vec4(1.0, 0.0, 0.0, 0.0)
    # Whether or not we need to compute the exact distance.
    get_dist = shrink_sphere
    # We can use GJK intersection algorithm only for collision detection if we do not have to compute the distance.
    backup_gjk = not get_dist
    # Support vector to compute the next support point.
    support_vector = gs.qd_vec3(0.0, 0.0, 0.0)
    support_vector_norm = gs.qd_float(0.0)
    # Whether or not the main loop finished early because intersection or seperation was detected.
    early_stop = False

    # Set initial guess of support vector using the thread-local positions, which should be a non-zero vector.
    approx_witness_point_obj1 = pos_a
    approx_witness_point_obj2 = pos_b
    support_vector = approx_witness_point_obj1 - approx_witness_point_obj2
    if support_vector.dot(support_vector) < collider_info.gjk.FLOAT_MIN_SQ[None]:
        support_vector = gs.qd_vec3(1.0, 0.0, 0.0)

    # Epsilon for convergence check.
    epsilon = gs.qd_float(0.0)
    if not func_is_discrete_geoms(i_ga, i_gb, dyn_info):
        # If the objects are smooth, finite convergence is not guaranteed, so we need to set some epsilon
        # to determine convergence.
        epsilon = 0.5 * (collider_info.gjk.tolerance[None] ** 2)

    for i in range(collider_info.gjk.gjk_max_iterations[None]):
        # Compute the current support points
        support_vector_norm = support_vector.norm()
        if support_vector_norm < collider_info.gjk.FLOAT_MIN[None]:
            # If the support vector is too small, it means that origin is located in the Minkowski difference
            # with high probability, so we can stop.
            break

        # Dir to compute the support point (pointing from obj1 to obj2)
        dir = -support_vector * (1.0 / support_vector_norm)

        (
            gjk_state.simplex_vertex.obj1[i_b, n],
            gjk_state.simplex_vertex.obj2[i_b, n],
            gjk_state.simplex_vertex.local_obj1[i_b, n],
            gjk_state.simplex_vertex.local_obj2[i_b, n],
            gjk_state.simplex_vertex.id1[i_b, n],
            gjk_state.simplex_vertex.id2[i_b, n],
            gjk_state.simplex_vertex.mink[i_b, n],
        ) = func_support(
            i_ga,
            i_gb,
            i_b,
            dir,
            pos_a,
            quat_a,
            pos_b,
            quat_b,
            shrink_sphere,
            collider_state,
            gjk_state,
            dyn_info,
            collider_info,
            rigid_config,
            collider_static_config,
        )

        # Early stopping based on Frank-Wolfe duality gap. We need to find the minimum [support_vector_norm],
        # and if we denote it as [x], the problem formulation is: min_x |x|^2.
        # If we denote f(x) = |x|^2, then the Frank-Wolfe duality gap is:
        # |x - x_min|^2 <= < grad f(x), x - s> = < 2x, x - s >,
        # where s is the vertex of the Minkowski difference found by x. Here < 2x, x - s > is guaranteed to be
        # non-negative, and 2 is cancelled out in the definition of the epsilon.
        x_k = support_vector
        s_k = gjk_state.simplex_vertex.mink[i_b, n]
        diff = x_k - s_k
        if diff.dot(x_k) < epsilon:
            # Convergence condition is met, we can stop.
            if i == 0:
                n = 1
            break

        # Check if the objects are separated using support vector
        if not get_dist:
            is_separated = x_k.dot(s_k) > 0.0
            if is_separated:
                nsimplex = 0
                nx = 0
                dist = collider_info.gjk.FLOAT_MAX[None]
                early_stop = True
                break

        if n == 3 and backup_gjk:
            # Tetrahedron is generated, try to detect collision if possible.
            intersect_code = func_gjk_intersect(
                i_ga,
                i_gb,
                i_b,
                pos_a,
                quat_a,
                pos_b,
                quat_b,
                collider_state,
                gjk_state,
                dyn_info,
                collider_info,
                rigid_config,
                collider_static_config,
            )
            if intersect_code == GJK_RETURN_CODE.SEPARATED:
                # No intersection, objects are separated
                nx = 0
                dist = collider_info.gjk.FLOAT_MAX[None]
                nsimplex = 0
                early_stop = True
                break
            elif intersect_code == GJK_RETURN_CODE.INTERSECT:
                # Intersection found
                nx = 0
                dist = 0.0
                nsimplex = 4
                early_stop = True
                break
            else:
                # Since gjk_intersect failed (e.g. origin is on the simplex face), fallback to distance computation
                backup_gjk = False

        # Compute the barycentric coordinates of the closest point to the origin in the simplex
        _lambda = func_gjk_subdistance(i_b, n + 1, gjk_state, collider_info)

        # Remove vertices from the simplex with zero barycentric coordinates
        n = 0
        for j in qd.static(range(4)):
            if _lambda[j] > 0:
                gjk_state.simplex_vertex.obj1[i_b, n] = gjk_state.simplex_vertex.obj1[i_b, j]
                gjk_state.simplex_vertex.obj2[i_b, n] = gjk_state.simplex_vertex.obj2[i_b, j]
                gjk_state.simplex_vertex.id1[i_b, n] = gjk_state.simplex_vertex.id1[i_b, j]
                gjk_state.simplex_vertex.id2[i_b, n] = gjk_state.simplex_vertex.id2[i_b, j]
                gjk_state.simplex_vertex.mink[i_b, n] = gjk_state.simplex_vertex.mink[i_b, j]
                _lambda[n] = _lambda[j]
                n += 1

        # Should not occur
        if n < 1:
            nsimplex = 0
            nx = 0
            dist = collider_info.gjk.FLOAT_MAX[None]
            early_stop = True
            break

        # Get the next support vector
        next_support_vector = func_simplex_vertex_linear_comb(i_b, 2, 0, 1, 2, 3, _lambda, n, gjk_state)
        if func_is_equal_vec(next_support_vector, support_vector, collider_info.gjk.FLOAT_MIN[None]):
            # If the next support vector is equal to the previous one, we converged to the minimum distance
            break

        support_vector = next_support_vector

        if n == 4:
            # We have a tetrahedron containing the origin, so we can return early. This is because only when
            # the origin is inside the tetrahedron, the barycentric coordinates are all positive. While MuJoCo
            # does not set the [support_vector_norm] to zero as we do, it is necessary, because otherwise the
            # [support_vector_norm] could be non-zero value even if there is contact.
            support_vector_norm = 0
            break

    if not early_stop:
        # If [get_dist] was True and there was no numerical error, [return_code] would be SUCCESS.
        nx = 1
        nsimplex = n
        dist = support_vector_norm

        # Compute witness points
        for i in range(2):
            witness_point = func_simplex_vertex_linear_comb(i_b, i, 0, 1, 2, 3, _lambda, nsimplex, gjk_state)
            if i == 0:
                gjk_state.witness.point_obj1[i_b, 0] = witness_point
            else:
                gjk_state.witness.point_obj2[i_b, 0] = witness_point

    gjk_state.n_witness[i_b] = nx
    gjk_state.distance[i_b] = dist
    gjk_state.nsimplex[i_b] = nsimplex

    return gjk_state.distance[i_b]


@qd.func
def func_gjk_intersect(
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
    Check if the two objects intersect using the GJK algorithm.

    This function refines the simplex until it contains the origin or it is determined that the objects are
    separated. It is used to check if the objects intersect, not to find the minimum distance between them.
    """
    # Copy simplex to temporary storage
    for i in qd.static(range(4)):
        gjk_state.simplex_vertex_intersect.obj1[i_b, i] = gjk_state.simplex_vertex.obj1[i_b, i]
        gjk_state.simplex_vertex_intersect.obj2[i_b, i] = gjk_state.simplex_vertex.obj2[i_b, i]
        gjk_state.simplex_vertex_intersect.id1[i_b, i] = gjk_state.simplex_vertex.id1[i_b, i]
        gjk_state.simplex_vertex_intersect.id2[i_b, i] = gjk_state.simplex_vertex.id2[i_b, i]
        gjk_state.simplex_vertex_intersect.mink[i_b, i] = gjk_state.simplex_vertex.mink[i_b, i]

    # Simplex index
    si = qd.Vector([0, 1, 2, 3], dt=gs.qd_int)

    flag = GJK_RETURN_CODE.NUM_ERROR
    for i in range(collider_info.gjk.gjk_max_iterations[None]):
        # Compute normal and signed distance of the triangle faces of the simplex with respect to the origin.
        # These normals are supposed to point outwards from the simplex.
        # If the origin is inside the plane, [sdist] will be positive.
        is_sdist_all_zero = True
        for j in range(4):
            s0, s1, s2 = si[2], si[1], si[3]
            if j == 1:
                s0, s1, s2 = si[0], si[2], si[3]
            elif j == 2:
                s0, s1, s2 = si[1], si[0], si[3]
            elif j == 3:
                s0, s1, s2 = si[0], si[1], si[2]

            n, s = func_gjk_triangle_info(i_b, s0, s1, s2, gjk_state, collider_info)

            gjk_state.simplex_buffer_intersect.normal[i_b, j] = n
            gjk_state.simplex_buffer_intersect.sdist[i_b, j] = s

            if qd.abs(s) > collider_info.gjk.FLOAT_MIN[None]:
                is_sdist_all_zero = False

        # If the origin is strictly on any affine hull of the faces, convergence will fail, so ignore this case
        if is_sdist_all_zero:
            break

        # Find the face with the smallest signed distance. We need to find [min_i] for the next iteration.
        min_i = 0
        for j in qd.static(range(1, 4)):
            if gjk_state.simplex_buffer_intersect.sdist[i_b, j] < gjk_state.simplex_buffer_intersect.sdist[i_b, min_i]:
                min_i = j

        min_si = si[min_i]
        min_normal = gjk_state.simplex_buffer_intersect.normal[i_b, min_i]
        min_sdist = gjk_state.simplex_buffer_intersect.sdist[i_b, min_i]

        # If origin is inside the simplex, the signed distances will all be positive
        if min_sdist >= 0:
            # Origin is inside the simplex, so we can stop
            flag = GJK_RETURN_CODE.INTERSECT

            # Copy the temporary simplex to the main simplex
            for j in qd.static(range(4)):
                gjk_state.simplex_vertex.obj1[i_b, j] = gjk_state.simplex_vertex_intersect.obj1[i_b, si[j]]
                gjk_state.simplex_vertex.obj2[i_b, j] = gjk_state.simplex_vertex_intersect.obj2[i_b, si[j]]
                gjk_state.simplex_vertex.id1[i_b, j] = gjk_state.simplex_vertex_intersect.id1[i_b, si[j]]
                gjk_state.simplex_vertex.id2[i_b, j] = gjk_state.simplex_vertex_intersect.id2[i_b, si[j]]
                gjk_state.simplex_vertex.mink[i_b, j] = gjk_state.simplex_vertex_intersect.mink[i_b, si[j]]
            break

        # Replace the worst vertex (which has the smallest signed distance) with new candidate
        (
            gjk_state.simplex_vertex_intersect.obj1[i_b, min_si],
            gjk_state.simplex_vertex_intersect.obj2[i_b, min_si],
            gjk_state.simplex_vertex_intersect.local_obj1[i_b, min_si],
            gjk_state.simplex_vertex_intersect.local_obj2[i_b, min_si],
            gjk_state.simplex_vertex_intersect.id1[i_b, min_si],
            gjk_state.simplex_vertex_intersect.id2[i_b, min_si],
            gjk_state.simplex_vertex_intersect.mink[i_b, min_si],
        ) = func_support(
            i_ga,
            i_gb,
            i_b,
            min_normal,
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

        # Check if the origin is strictly outside of the Minkowski difference (which means there is no collision)
        new_minkowski = gjk_state.simplex_vertex_intersect.mink[i_b, min_si]

        is_no_collision = new_minkowski.dot(min_normal) < 0
        if is_no_collision:
            flag = GJK_RETURN_CODE.SEPARATED
            break

        # Swap vertices in the simplex to retain orientation
        m = (min_i + 1) % 4
        n = (min_i + 2) % 4
        swap = si[m]
        si[m] = si[n]
        si[n] = swap

    return flag


@qd.func
def func_gjk_triangle_info(
    i_b, i_va, i_vb, i_vc, gjk_state: array_class.GJKState, collider_info: array_class.ColliderInfo
):
    """
    Compute normal and signed distance of the triangle face on the simplex from the origin.
    """
    vertex_1 = gjk_state.simplex_vertex_intersect.mink[i_b, i_va]
    vertex_2 = gjk_state.simplex_vertex_intersect.mink[i_b, i_vb]
    vertex_3 = gjk_state.simplex_vertex_intersect.mink[i_b, i_vc]

    normal = (vertex_3 - vertex_1).cross(vertex_2 - vertex_1)
    normal_length = normal.norm()

    sdist = 0.0
    if (normal_length > collider_info.gjk.FLOAT_MIN[None]) and (normal_length < collider_info.gjk.FLOAT_MAX[None]):
        normal = normal * (1.0 / normal_length)
        sdist = normal.dot(vertex_1)
    else:
        # If the normal length is unstable, return max distance.
        sdist = collider_info.gjk.FLOAT_MAX[None]

    return normal, sdist


@qd.func
def func_gjk_subdistance(i_b, n, gjk_state: array_class.GJKState, collider_info: array_class.ColliderInfo):
    """
    Compute the barycentric coordinates of the closest point to the origin in the n-simplex.

    .. seealso::
    Montanari, Mattia, Nik Petrinic, and Ettore Barbieri. "Improving the GJK algorithm for faster and more reliable
    distance queries between convex objects." ACM Transactions on Graphics (TOG) 36.3 (2017): 1-17.
    https://dl.acm.org/doi/10.1145/3072959.3083724
    """
    _lambda = qd.math.vec4(1.0, 0.0, 0.0, 0.0)

    # Whether or not the subdistance was computed successfully for the n-simplex.
    flag = RETURN_CODE.SUCCESS

    dmin = collider_info.gjk.FLOAT_MAX[None]

    if n == 4:
        _lambda, flag3d = func_gjk_subdistance_3d(i_b, 0, 1, 2, 3, gjk_state)
        flag = flag3d

    if (flag == RETURN_CODE.FAIL) or n == 3:
        failed_3d = n == 4
        num_iter = 1
        if failed_3d:
            # Iterate through 4 faces of the tetrahedron
            num_iter = 4

        for i in range(num_iter):
            k_1, k_2, k_3 = i, (i + 1) % 4, (i + 2) % 4
            _lambda2d, flag2d = func_gjk_subdistance_2d(i_b, k_1, k_2, k_3, gjk_state, collider_info)

            if failed_3d:
                if flag2d == RETURN_CODE.SUCCESS:
                    closest_point = func_simplex_vertex_linear_comb(i_b, 2, k_1, k_2, k_3, 0, _lambda2d, 3, gjk_state)
                    d = closest_point.dot(closest_point)
                    if d < dmin:
                        dmin = d
                        _lambda.fill(0.0)
                        _lambda[k_1] = _lambda2d[0]
                        _lambda[k_2] = _lambda2d[1]
                        _lambda[k_3] = _lambda2d[2]
            else:
                if flag2d == RETURN_CODE.SUCCESS:
                    _lambda = _lambda2d
                flag = flag2d

    if (flag == RETURN_CODE.FAIL) or n == 2:
        failed_3d = n == 4
        failed_2d = n == 3

        num_iter = 1
        if failed_3d:
            # Iterate through 6 edges of the tetrahedron
            num_iter = 6
        elif failed_2d:
            # Iterate through 3 edges of the triangle
            num_iter = 3

        for i in range(num_iter):
            k_1, k_2 = i, (i + 1) % 3
            if i >= 3:
                k_1, k_2 = i - 3, 3

            _lambda1d = func_gjk_subdistance_1d(i_b, k_1, k_2, gjk_state)

            if failed_3d or failed_2d:
                closest_point = func_simplex_vertex_linear_comb(i_b, 2, k_1, k_2, 0, 0, _lambda1d, 2, gjk_state)
                d = closest_point.dot(closest_point)
                if d < dmin:
                    dmin = d
                    _lambda.fill(0.0)
                    _lambda[k_1] = _lambda1d[0]
                    _lambda[k_2] = _lambda1d[1]
            else:
                _lambda = _lambda1d

    return _lambda


@qd.func
def func_gjk_subdistance_3d(i_b, i_s1, i_s2, i_s3, i_s4, gjk_state: array_class.GJKState):
    """
    Compute the barycentric coordinates of the closest point to the origin in the 3-simplex (tetrahedron).
    """
    flag = RETURN_CODE.FAIL
    _lambda = gs.qd_vec4(0, 0, 0, 0)

    # Simplex vertices
    s1 = gjk_state.simplex_vertex.mink[i_b, i_s1]
    s2 = gjk_state.simplex_vertex.mink[i_b, i_s2]
    s3 = gjk_state.simplex_vertex.mink[i_b, i_s3]
    s4 = gjk_state.simplex_vertex.mink[i_b, i_s4]

    # Compute the cofactors to find det(M), which corresponds to the signed volume of the tetrahedron
    Cs = qd.math.vec4(0.0, 0.0, 0.0, 0.0)
    for i in range(4):
        v1, v2, v3 = s2, s3, s4
        if i == 1:
            v1, v2, v3 = s1, s3, s4
        elif i == 2:
            v1, v2, v3 = s1, s2, s4
        elif i == 3:
            v1, v2, v3 = s1, s2, s3
        Cs[i] = func_det3(v1, v2, v3)
    Cs[0], Cs[2] = -Cs[0], -Cs[2]
    m_det = Cs.sum()

    # Compare sign of the cofactors with the determinant
    scs = gs.qd_ivec4(0, 0, 0, 0)
    for i in range(4):
        scs[i] = func_compare_sign(Cs[i], m_det)

    if scs.all():
        # If all barycentric coordinates are positive, the origin is inside the tetrahedron
        _lambda = Cs / m_det
        flag = RETURN_CODE.SUCCESS

    return _lambda, flag


@qd.func
def func_gjk_subdistance_2d(
    i_b, i_s1, i_s2, i_s3, gjk_state: array_class.GJKState, collider_info: array_class.ColliderInfo
):
    """
    Compute the barycentric coordinates of the closest point to the origin in the 2-simplex (triangle).
    """
    _lambda = qd.math.vec4(0, 0, 0, 0)
    flag = RETURN_CODE.FAIL

    # Project origin onto affine hull of the simplex (triangle)
    proj_orig, proj_flag = func_project_origin_to_plane(
        gjk_state.simplex_vertex.mink[i_b, i_s1],
        gjk_state.simplex_vertex.mink[i_b, i_s2],
        gjk_state.simplex_vertex.mink[i_b, i_s3],
        collider_info,
    )

    if proj_flag == RETURN_CODE.SUCCESS:
        # We should find the barycentric coordinates of the projected point, but the linear system is not square:
        # [ s1.x, s2.x, s3.x ] [ l1 ] = [ proj_o.x ]
        # [ s1.y, s2.y, s3.y ] [ l2 ] = [ proj_o.y ]
        # [ s1.z, s2.z, s3.z ] [ l3 ] = [ proj_o.z ]
        # [ 1,    1,    1,   ] [ ?  ] = [ 1.0 ]
        # So we remove one row before solving the system. We exclude the axis with the largest projection of the
        # simplex using the minors of the above linear system.
        s1 = gjk_state.simplex_vertex.mink[i_b, i_s1]
        s2 = gjk_state.simplex_vertex.mink[i_b, i_s2]
        s3 = gjk_state.simplex_vertex.mink[i_b, i_s3]

        ms = gs.qd_vec3(
            s2[1] * s3[2] - s2[2] * s3[1] - s1[1] * s3[2] + s1[2] * s3[1] + s1[1] * s2[2] - s1[2] * s2[1],
            s2[0] * s3[2] - s2[2] * s3[0] - s1[0] * s3[2] + s1[2] * s3[0] + s1[0] * s2[2] - s1[2] * s2[0],
            s2[0] * s3[1] - s2[1] * s3[0] - s1[0] * s3[1] + s1[1] * s3[0] + s1[0] * s2[1] - s1[1] * s2[0],
        )
        absms = qd.abs(ms)

        m_max = 0.0
        s1_2d, s2_2d, s3_2d = gs.qd_vec2(0, 0), gs.qd_vec2(0, 0), gs.qd_vec2(0, 0)
        proj_orig_2d = gs.qd_vec2(0, 0)

        for i in range(3):
            if absms[i] >= absms[(i + 1) % 3] and absms[i] >= absms[(i + 2) % 3]:
                # Remove the i-th row from the linear system
                m_max = ms[i]

                i0, i1 = (i + 1) % 3, (i + 2) % 3
                if i == 1:
                    i0, i1 = i1, i0

                s1_2d[0], s1_2d[1] = s1[i0], s1[i1]
                s2_2d[0], s2_2d[1] = s2[i0], s2[i1]
                s3_2d[0], s3_2d[1] = s3[i0], s3[i1]
                proj_orig_2d[0] = proj_orig[i0]
                proj_orig_2d[1] = proj_orig[i1]
                break

        # Now we find the barycentric coordinates of the projected point by solving the linear system:
        # [ s1_2d.x, s2_2d.x, s3_2d.x ] [ l1 ] = [ proj_orig_2d.x ]
        # [ s1_2d.y, s2_2d.y, s3_2d.y ] [ l2 ] = [ proj_orig_2d.y ]
        # [ 1,       1,       1,      ] [ l3 ] = [ 1.0 ]
        cs = gs.qd_vec3(0, 0, 0)
        for i in range(3):
            s2d0, s2d1 = s2_2d, s3_2d
            if i == 1:
                s2d0, s2d1 = s3_2d, s1_2d
            elif i == 2:
                s2d0, s2d1 = s1_2d, s2_2d
            # Corresponds to the signed area of 2-simplex (triangle): (proj_orig_2d, s2d0, s2d1)
            cs[i] = (
                proj_orig_2d[0] * s2d0[1]
                + proj_orig_2d[1] * s2d1[0]
                + s2d0[0] * s2d1[1]
                - proj_orig_2d[0] * s2d1[1]
                - proj_orig_2d[1] * s2d0[0]
                - s2d1[0] * s2d0[1]
            )

        # Compare sign of the cofactors with the determinant
        scs = gs.qd_ivec3(0, 0, 0)
        for i in range(3):
            scs[i] = func_compare_sign(cs[i], m_max)

        if scs.all():
            # If all barycentric coordinates are positive, the origin is inside the 2-simplex (triangle)
            for i in qd.static(range(3)):
                _lambda[i] = cs[i] / m_max
            flag = RETURN_CODE.SUCCESS

    return _lambda, flag


@qd.func
def func_gjk_subdistance_1d(i_b, i_s1, i_s2, gjk_state: array_class.GJKState):
    """
    Compute the barycentric coordinates of the closest point to the origin in the 1-simplex (line segment).
    """
    _lambda = gs.qd_vec4(0, 0, 0, 0)

    s1 = gjk_state.simplex_vertex.mink[i_b, i_s1]
    s2 = gjk_state.simplex_vertex.mink[i_b, i_s2]
    p_o = func_project_origin_to_line(s1, s2)

    mu_max = 0.0
    index = -1
    for i in range(3):
        mu = s1[i] - s2[i]
        if qd.abs(mu) >= qd.abs(mu_max):
            mu_max = mu
            index = i

    C1 = p_o[index] - s2[index]
    C2 = s1[index] - p_o[index]

    # Determine if projection of origin lies inside 1-simplex
    if func_compare_sign(mu_max, C1) and func_compare_sign(mu_max, C2):
        _lambda[0] = C1 / mu_max
        _lambda[1] = C2 / mu_max
    else:
        _lambda[0] = 0.0
        _lambda[1] = 1.0

    return _lambda


@qd.func
def func_is_sphere_swept_geom(i_g, dyn_info: array_class.DynInfo):
    """
    Check if the given geoms are sphere-swept geometries.
    """
    geom_type = dyn_info.geoms.type[i_g]
    return geom_type == gs.GEOM_TYPE.SPHERE or geom_type == gs.GEOM_TYPE.CAPSULE


@qd.func
def func_project_origin_to_line(v1, v2):
    """
    Project the origin onto the line defined by the simplex vertices.

    P = v2 - ((v1 * diff) / (diff * diff)) * diff
    """
    diff = v2 - v1
    k = v2.dot(diff) / diff.dot(diff)
    P = v2 - k * diff

    return P


@qd.func
def func_simplex_vertex_linear_comb(i_b, i_v, i_s1, i_s2, i_s3, i_s4, _lambda, n, gjk_state: array_class.GJKState):
    """
    Compute the linear combination of the simplex vertices

    Parameters:
    ----------
    i_v: int
        Which vertex to use (0: obj1, 1: obj2, 2: minkowski)
    n: int
        Number of vertices to combine, combine the first n vertices
    """
    res = gs.qd_vec3(0, 0, 0)

    s1 = gjk_state.simplex_vertex.obj1[i_b, i_s1]
    s2 = gjk_state.simplex_vertex.obj1[i_b, i_s2]
    s3 = gjk_state.simplex_vertex.obj1[i_b, i_s3]
    s4 = gjk_state.simplex_vertex.obj1[i_b, i_s4]
    if i_v == 1:
        s1 = gjk_state.simplex_vertex.obj2[i_b, i_s1]
        s2 = gjk_state.simplex_vertex.obj2[i_b, i_s2]
        s3 = gjk_state.simplex_vertex.obj2[i_b, i_s3]
        s4 = gjk_state.simplex_vertex.obj2[i_b, i_s4]
    elif i_v == 2:
        s1 = gjk_state.simplex_vertex.mink[i_b, i_s1]
        s2 = gjk_state.simplex_vertex.mink[i_b, i_s2]
        s3 = gjk_state.simplex_vertex.mink[i_b, i_s3]
        s4 = gjk_state.simplex_vertex.mink[i_b, i_s4]

    c1 = _lambda[0]
    c2 = _lambda[1]
    c3 = _lambda[2]
    c4 = _lambda[3]

    if n == 1:
        res = s1 * c1
    elif n == 2:
        res = s1 * c1 + s2 * c2
    elif n == 3:
        res = s1 * c1 + s2 * c2 + s3 * c3
    else:
        res = s1 * c1 + s2 * c2 + s3 * c3 + s4 * c4
    return res


@qd.func
def func_safe_gjk(
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
    Safe GJK algorithm to compute the minimum distance between two convex objects.
    using thread-local pos/quat for both geometries.

    Thread-safety note: Geometry indices `i_ga` and `i_gb` are only used for read-only
    metadata access (checking geometry types via `func_is_discrete_geoms`) and passing to
    support functions. They do not access `geoms_state.pos` or `geoms_state.quat`.

    This implementation is safer than the one based on the MuJoCo implementation for the following reasons:
    1) It guarantees that the origin is strictly inside the tetrahedron when the intersection is detected.
    2) It guarantees to generate a non-degenerate tetrahedron if there is no numerical error, which is necessary
    for the following EPA algorithm to work correctly.
    3) When computing the face normals on the simplex, it uses a more robust method than using the origin.

    TODO: This implementation could be improved by using shrink_sphere option as the MuJoCo implementation does.
    TODO: This implementation could be further improved by referencing the follow-up work shown below.

    .. seealso::
    Original paper:
    Gilbert, Elmer G., Daniel W. Johnson, and S. Sathiya Keerthi.
    "A fast procedure for computing the distance between complex objects in three-dimensional space."
    IEEE Journal on Robotics and Automation 4.2 (2002): 193-203.

    Further improvements:
    Cameron, Stephen. "Enhancing GJK: Computing minimum and penetration distances between convex polyhedra."
    Proceedings of international conference on robotics and automation. Vol. 4. IEEE, 1997.
    https://www.cs.ox.ac.uk/people/stephen.cameron/distances/gjk2.4/

    Montaut, Louis, et al. "Collision detection accelerated: An optimization perspective."
    https://arxiv.org/abs/2205.09663
    """
    # Compute the initial tetrahedron from 4 fixed-direction support queries
    init_flag = RETURN_CODE.SUCCESS
    gjk_state.simplex.nverts[i_b] = 0
    for i in range(4):
        dir = qd.Vector.zero(gs.qd_float, 3)
        dir[2 - i // 2] = 1.0 - 2.0 * (i % 2)

        obj1, obj2, local_obj1, local_obj2, id1, id2, minkowski = func_safe_gjk_support(
            i_ga,
            i_gb,
            i_b,
            pos_a,
            quat_a,
            pos_b,
            quat_b,
            dir,
            collider_state,
            gjk_state,
            dyn_info,
            rigid_info,
            collider_info,
            rigid_config,
            collider_static_config,
        )

        # Check if the new vertex would make a valid simplex.
        valid = func_is_new_simplex_vertex_valid(i_b, id1, id2, minkowski, gjk_state, collider_info)

        # If this is not a valid vertex, fall back to a brute-force routine to find a valid vertex.
        if not valid:
            obj1, obj2, local_obj1, local_obj2, id1, id2, minkowski, init_flag = func_search_valid_simplex_vertex(
                i_ga,
                i_gb,
                i_b,
                pos_a,
                quat_a,
                pos_b,
                quat_b,
                collider_state,
                gjk_state,
                dyn_info,
                rigid_info,
                collider_info,
                rigid_config,
                collider_static_config,
            )
            # If the brute-force search failed, we cannot proceed with GJK.
            if init_flag == RETURN_CODE.FAIL:
                break

        gjk_state.simplex_vertex.obj1[i_b, i] = obj1
        gjk_state.simplex_vertex.obj2[i_b, i] = obj2
        gjk_state.simplex_vertex.local_obj1[i_b, i] = local_obj1
        gjk_state.simplex_vertex.local_obj2[i_b, i] = local_obj2
        gjk_state.simplex_vertex.id1[i_b, i] = id1
        gjk_state.simplex_vertex.id2[i_b, i] = id2
        gjk_state.simplex_vertex.mink[i_b, i] = minkowski
        gjk_state.simplex.nverts[i_b] += 1

    gjk_flag = GJK_RETURN_CODE.SEPARATED
    if init_flag == RETURN_CODE.SUCCESS:
        # Simplex index
        si = qd.Vector([0, 1, 2, 3], dt=gs.qd_int)

        for i in range(collider_info.gjk.gjk_max_iterations[None]):
            # Compute normal and signed distance of the triangle faces of the simplex with respect to the origin.
            # These normals are supposed to point outwards from the simplex. If the origin is inside the plane,
            # [sdist] will be positive.
            for j in range(4):
                s0, s1, s2, ap = si[2], si[1], si[3], si[0]
                if j == 1:
                    s0, s1, s2, ap = si[0], si[2], si[3], si[1]
                elif j == 2:
                    s0, s1, s2, ap = si[1], si[0], si[3], si[2]
                elif j == 3:
                    s0, s1, s2, ap = si[0], si[1], si[2], si[3]

                n, s = func_safe_gjk_triangle_info(i_b, s0, s1, s2, ap, gjk_state)

                gjk_state.simplex_buffer.normal[i_b, j] = n
                gjk_state.simplex_buffer.sdist[i_b, j] = s

            # Find the face with the smallest signed distance. We need to find [min_i] for the next iteration.
            min_i = 0
            for j in qd.static(range(1, 4)):
                if gjk_state.simplex_buffer.sdist[i_b, j] < gjk_state.simplex_buffer.sdist[i_b, min_i]:
                    min_i = j

            min_si = si[min_i]
            min_normal = gjk_state.simplex_buffer.normal[i_b, min_i]
            min_sdist = gjk_state.simplex_buffer.sdist[i_b, min_i]

            # If origin is inside the simplex, the signed distances will all be positive
            if min_sdist >= 0:
                # Origin is inside the simplex, so we can stop
                gjk_flag = GJK_RETURN_CODE.INTERSECT
                break

            # Check if the new vertex would make a valid simplex.
            gjk_state.simplex.nverts[i_b] = 3
            if min_si != 3:
                gjk_state.simplex_vertex.obj1[i_b, min_si] = gjk_state.simplex_vertex.obj1[i_b, 3]
                gjk_state.simplex_vertex.obj2[i_b, min_si] = gjk_state.simplex_vertex.obj2[i_b, 3]
                gjk_state.simplex_vertex.local_obj1[i_b, min_si] = gjk_state.simplex_vertex.local_obj1[i_b, 3]
                gjk_state.simplex_vertex.local_obj2[i_b, min_si] = gjk_state.simplex_vertex.local_obj2[i_b, 3]
                gjk_state.simplex_vertex.id1[i_b, min_si] = gjk_state.simplex_vertex.id1[i_b, 3]
                gjk_state.simplex_vertex.id2[i_b, min_si] = gjk_state.simplex_vertex.id2[i_b, 3]
                gjk_state.simplex_vertex.mink[i_b, min_si] = gjk_state.simplex_vertex.mink[i_b, 3]

            # Find a new candidate vertex to replace the worst vertex (which has the smallest signed distance)
            obj1, obj2, local_obj1, local_obj2, id1, id2, minkowski = func_safe_gjk_support(
                i_ga,
                i_gb,
                i_b,
                pos_a,
                quat_a,
                pos_b,
                quat_b,
                min_normal,
                collider_state,
                gjk_state,
                dyn_info,
                rigid_info,
                collider_info,
                rigid_config,
                collider_static_config,
            )

            duplicate = func_is_new_simplex_vertex_duplicate(i_b, id1, id2, gjk_state)
            if duplicate:
                # If the new vertex is a duplicate, it means separation.
                gjk_flag = GJK_RETURN_CODE.SEPARATED
                break

            degenerate = func_is_new_simplex_vertex_degenerate(i_b, minkowski, gjk_state, collider_info)
            if degenerate:
                # If the new vertex is degenerate, we cannot proceed with GJK.
                gjk_flag = GJK_RETURN_CODE.NUM_ERROR
                break

            # Check if the origin is strictly outside of the Minkowski difference (which means there is no collision)
            is_no_collision = minkowski.dot(min_normal) < 0.0
            if is_no_collision:
                gjk_flag = GJK_RETURN_CODE.SEPARATED
                break

            gjk_state.simplex_vertex.obj1[i_b, 3] = obj1
            gjk_state.simplex_vertex.obj2[i_b, 3] = obj2
            gjk_state.simplex_vertex.local_obj1[i_b, 3] = local_obj1
            gjk_state.simplex_vertex.local_obj2[i_b, 3] = local_obj2
            gjk_state.simplex_vertex.id1[i_b, 3] = id1
            gjk_state.simplex_vertex.id2[i_b, 3] = id2
            gjk_state.simplex_vertex.mink[i_b, 3] = minkowski
            gjk_state.simplex.nverts[i_b] = 4

    if gjk_flag == GJK_RETURN_CODE.INTERSECT:
        gjk_state.distance[i_b] = 0.0
    else:
        gjk_flag = GJK_RETURN_CODE.SEPARATED
        gjk_state.distance[i_b] = collider_info.gjk.FLOAT_MAX[None]

    return gjk_flag


@qd.func
def func_is_new_simplex_vertex_valid(
    i_b, id1, id2, mink, gjk_state: array_class.GJKState, collider_info: array_class.ColliderInfo
):
    """
    Check validity of the incoming simplex vertex (defined by id1, id2 and mink).

    To be a new valid simplex vertex, it should satisfy the following conditions:
    1) The vertex should not be already in the simplex.
    2) The simplex should not be degenerate after insertion.
    """
    return (not func_is_new_simplex_vertex_duplicate(i_b, id1, id2, gjk_state)) and (
        not func_is_new_simplex_vertex_degenerate(i_b, mink, gjk_state, collider_info)
    )


@qd.func
def func_is_new_simplex_vertex_duplicate(i_b, id1, id2, gjk_state: array_class.GJKState):
    """
    Check if the incoming simplex vertex is already in the simplex.
    """
    nverts = gjk_state.simplex.nverts[i_b]
    found = False
    for i in range(nverts):
        if id1 == -1 or (gjk_state.simplex_vertex.id1[i_b, i] != id1):
            continue
        if id2 == -1 or (gjk_state.simplex_vertex.id2[i_b, i] != id2):
            continue
        found = True
        break
    return found


@qd.func
def func_is_new_simplex_vertex_degenerate(
    i_b, mink, gjk_state: array_class.GJKState, collider_info: array_class.ColliderInfo
):
    """
    Check if the simplex becomes degenerate after inserting a new vertex, assuming that the current simplex is okay.
    """
    is_degenerate = False

    # Check if the new vertex is not very close to the existing vertices
    nverts = gjk_state.simplex.nverts[i_b]
    for i in range(nverts):
        if (gjk_state.simplex_vertex.mink[i_b, i] - mink).norm_sqr() < (
            collider_info.gjk.simplex_max_degeneracy_sq[None]
        ):
            is_degenerate = True
            break

    if not is_degenerate:
        # Check the validity based on the simplex dimension
        if nverts == 2:
            # Becomes a triangle if valid, check if the three vertices are not collinear
            is_degenerate = func_is_colinear(
                gjk_state.simplex_vertex.mink[i_b, 0], gjk_state.simplex_vertex.mink[i_b, 1], mink, collider_info
            )
        elif nverts == 3:
            # Becomes a tetrahedron if valid, check if the four vertices are not coplanar
            is_degenerate = func_is_coplanar(
                gjk_state.simplex_vertex.mink[i_b, 0],
                gjk_state.simplex_vertex.mink[i_b, 1],
                gjk_state.simplex_vertex.mink[i_b, 2],
                mink,
                collider_info,
            )

    return is_degenerate


@qd.func
def func_is_colinear(v1, v2, v3, collider_info: array_class.ColliderInfo):
    """
    Check if three points are collinear.

    This function assumes that every pair of points is non-degenerate, i.e. no pair of points is identical.
    """
    e1 = v2 - v1
    e2 = v3 - v1
    normal = e1.cross(e2)
    return normal.norm_sqr() < (collider_info.gjk.simplex_max_degeneracy_sq[None]) * e1.norm_sqr() * e2.norm_sqr()


@qd.func
def func_is_coplanar(v1, v2, v3, v4, collider_info: array_class.ColliderInfo):
    """
    Check if four points are coplanar.

    This function assumes that every triplet of points is non-degenerate, i.e. no triplet of points is collinear.
    """
    e1 = (v2 - v1).normalized()
    e2 = (v3 - v1).normalized()
    normal = e1.cross(e2)
    diff = v4 - v1
    return (normal.dot(diff) ** 2) < (
        collider_info.gjk.simplex_max_degeneracy_sq[None]
    ) * normal.norm_sqr() * diff.norm_sqr()


@qd.func
def func_search_valid_simplex_vertex(
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
    Search for a valid simplex vertex (non-duplicate, non-degenerate) in the Minkowski difference.
    using thread-local pos/quat for both geometries.
    """
    obj1 = gs.qd_vec3(0.0, 0.0, 0.0)
    obj2 = gs.qd_vec3(0.0, 0.0, 0.0)
    local_obj1 = gs.qd_vec3(0.0, 0.0, 0.0)
    local_obj2 = gs.qd_vec3(0.0, 0.0, 0.0)
    id1 = -1
    id2 = -1
    minkowski = gs.qd_vec3(0.0, 0.0, 0.0)
    flag = RETURN_CODE.FAIL

    # If both geometries are discrete, we can use a brute-force search to find a valid simplex vertex.
    if func_is_discrete_geoms(i_ga, i_gb, dyn_info):
        geom_nverts = gs.qd_ivec2(0, 0)
        for i in range(2):
            geom_nverts[i] = func_num_discrete_geom_vertices(i_ga if i == 0 else i_gb, dyn_info)

        num_cases = geom_nverts[0] * geom_nverts[1]
        for k in range(num_cases):
            m = (k + gjk_state.last_searched_simplex_vertex_id[i_b]) % num_cases
            i = m // geom_nverts[1]
            j = m % geom_nverts[1]

            id1 = dyn_info.geoms.vert_start[i_ga] + i
            id2 = dyn_info.geoms.vert_start[i_gb] + j
            for p in range(2):
                obj, local_obj = func_get_discrete_geom_vertex(
                    i_ga if p == 0 else i_gb,
                    i if p == 0 else j,
                    pos_a if p == 0 else pos_b,
                    quat_a if p == 0 else quat_b,
                    dyn_info,
                )
                if p == 0:
                    obj1 = obj
                    local_obj1 = local_obj
                else:
                    obj2 = obj
                    local_obj2 = local_obj
            minkowski = obj1 - obj2

            # Check if the new vertex is valid
            if func_is_new_simplex_vertex_valid(i_b, id1, id2, minkowski, gjk_state, collider_info):
                flag = RETURN_CODE.SUCCESS
                # Update buffer
                gjk_state.last_searched_simplex_vertex_id[i_b] = (m + 1) % num_cases
                break
    else:
        # Try search direction based on the current simplex.
        nverts = gjk_state.simplex.nverts[i_b]
        if nverts == 3:
            # If we have a triangle, use its normal as the search direction.
            v1 = gjk_state.simplex_vertex.mink[i_b, 0]
            v2 = gjk_state.simplex_vertex.mink[i_b, 1]
            v3 = gjk_state.simplex_vertex.mink[i_b, 2]
            dir = (v3 - v1).cross(v2 - v1).normalized()

            for i in range(2):
                d = dir if i == 0 else -dir
                obj1, obj2, local_obj1, local_obj2, id1, id2, minkowski = func_safe_gjk_support(
                    i_ga,
                    i_gb,
                    i_b,
                    pos_a,
                    quat_a,
                    pos_b,
                    quat_b,
                    d,
                    collider_state,
                    gjk_state,
                    dyn_info,
                    rigid_info,
                    collider_info,
                    rigid_config,
                    collider_static_config,
                )

                # Check if the new vertex is valid
                if func_is_new_simplex_vertex_valid(i_b, id1, id2, minkowski, gjk_state, collider_info):
                    flag = RETURN_CODE.SUCCESS
                    break

    return obj1, obj2, local_obj1, local_obj2, id1, id2, minkowski, flag


@qd.func
def func_num_discrete_geom_vertices(i_g, dyn_info: array_class.DynInfo):
    """
    Count the number of discrete vertices in the geometry.
    """
    vert_start = dyn_info.geoms.vert_start[i_g]
    vert_end = dyn_info.geoms.vert_end[i_g]
    count = vert_end - vert_start
    return count


@qd.func
def func_get_discrete_geom_vertex(
    i_g, i_v, pos: qd.types.vector(3), quat: qd.types.vector(4), dyn_info: array_class.DynInfo
):
    """
    Get the discrete vertex of the geometry for the given index [i_v].
    """
    geom_type = dyn_info.geoms.type[i_g]

    # Get the vertex position in the local frame of the geometry.
    v_ = qd.Vector([0.0, 0.0, 0.0], dt=gs.qd_float)
    if geom_type == gs.GEOM_TYPE.BOX:
        # For the consistency with the [func_support_box] function of [SupportField] class, we handle the box
        # vertex positions in a different way than the general mesh.
        v_ = qd.Vector(
            [
                (1.0 if (i_v & 1 == 1) else -1.0) * dyn_info.geoms.data[i_g][0] * 0.5,
                (1.0 if (i_v & 2 == 2) else -1.0) * dyn_info.geoms.data[i_g][1] * 0.5,
                (1.0 if (i_v & 4 == 4) else -1.0) * dyn_info.geoms.data[i_g][2] * 0.5,
            ],
            dt=gs.qd_float,
        )
    elif geom_type == gs.GEOM_TYPE.MESH:
        vert_start = dyn_info.geoms.vert_start[i_g]
        v_ = dyn_info.verts.init_pos[vert_start + i_v]

    # Transform the vertex position to the world frame using thread-local pos/quat
    v = gu.qd_transform_by_trans_quat(v_, pos, quat)

    return v, v_


@qd.func
def func_safe_gjk_triangle_info(i_b, i_ta, i_tb, i_tc, i_apex, gjk_state: array_class.GJKState):
    """
    Compute normal and signed distance of the triangle face on the simplex from the origin.

    The triangle is defined by the vertices [i_ta], [i_tb], and [i_tc], and the apex is used to orient the triangle
    normal, so that it points outward from the simplex. Thus, if the origin is inside the simplex in terms of this
    triangle, the signed distance will be positive.
    """
    vertex_1 = gjk_state.simplex_vertex.mink[i_b, i_ta]
    vertex_2 = gjk_state.simplex_vertex.mink[i_b, i_tb]
    vertex_3 = gjk_state.simplex_vertex.mink[i_b, i_tc]
    apex_vertex = gjk_state.simplex_vertex.mink[i_b, i_apex]

    # This normal is guaranteed to be non-zero because we build the simplex avoiding degenerate vertices.
    normal = (vertex_3 - vertex_1).cross(vertex_2 - vertex_1).normalized()

    # Reorient the normal to point outward from the simplex
    if normal.dot(apex_vertex - vertex_1) > 0.0:
        normal = -normal

    # Compute the signed distance from the origin to the triangle plane
    sdist = normal.dot(vertex_1)

    return normal, sdist


@qd.func
def func_safe_gjk_support(
    i_ga,
    i_gb,
    i_b,
    pos_a: qd.types.vector(3),
    quat_a: qd.types.vector(4),
    pos_b: qd.types.vector(3),
    quat_b: qd.types.vector(4),
    dir,
    collider_state: array_class.ColliderState,
    gjk_state: array_class.GJKState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    collider_info: array_class.ColliderInfo,
    rigid_config: qd.template(),
    collider_static_config: qd.template(),
):
    """
    Find support points on the two objects using [dir] to use in the [safe_gjk] algorithm.
    Uses thread-local pos/quat for both geometries.

    This is a more robust version of the support function that finds only one pair of support points, because this
    function perturbs the support direction to find the best support points that guarantee non-degenerate simplex
    in the GJK algorithm.

    The support count (number of candidate support vertices tied for the maximum dot product) is only evaluated once
    for the original direction (i=0). For perturbed directions (i=1..8), we skip the count and go straight to
    computing support points and checking simplex validity. This is critical for performance because the count check
    is ineffective for perturbed directions: the support field is discretized on a spherical grid, and an EPS-scale
    perturbation maps to the same grid cells, yielding the same count every time. The previous approach recomputed
    count for all 9 directions, wasting 8 evaluations that always returned > 1 for meshes with flat faces, without
    ever using the perturbed support points.

    Parameters:
    ----------
    dir: gs.qd_vec3
        The unit direction in which to find the support points, from [ga] (obj 1) to [gb] (obj 2).
    """
    EPS = rigid_info.EPS[None]

    obj1 = gs.qd_vec3(0.0, 0.0, 0.0)
    obj2 = gs.qd_vec3(0.0, 0.0, 0.0)
    local_obj1 = gs.qd_vec3(0.0, 0.0, 0.0)
    local_obj2 = gs.qd_vec3(0.0, 0.0, 0.0)
    id1 = gs.qd_int(-1)
    id2 = gs.qd_int(-1)
    mink = obj1 - obj2

    for i in range(9):
        n_dir = dir
        if i > 0:
            j = i - 1
            n_dir[0] += -(1.0 - 2.0 * (j & 1)) * EPS
            n_dir[1] += -(1.0 - 2.0 * (j & 2)) * EPS
            n_dir[2] += -(1.0 - 2.0 * (j & 4)) * EPS

        # First order normalization based on Taylor series is accurate enough
        n_dir *= 2.0 - n_dir.dot(dir)

        # Use the current direction to find the support points.
        for j in range(2):
            d = n_dir if j == 0 else -n_dir
            i_g = i_ga if j == 0 else i_gb
            pos = pos_a if j == 0 else pos_b
            quat = quat_a if j == 0 else quat_b

            sp, local_sp, si = support_driver(
                i_g,
                i_b,
                j,
                d,
                pos,
                quat,
                shrink_sphere=False,
                collider_state=collider_state,
                gjk_state=gjk_state,
                dyn_info=dyn_info,
                collider_info=collider_info,
                rigid_config=rigid_config,
                collider_static_config=collider_static_config,
            )
            if j == 0:
                obj1 = sp
                local_obj1 = local_sp
                id1 = si
            else:
                obj2 = sp
                local_obj2 = local_sp
                id2 = si

        mink = obj1 - obj2

        if i == 0:
            # Only check support count for the original direction. If unique, we are done.
            num_supports = func_count_support(i_ga, i_gb, quat_a, quat_b, n_dir, dyn_info, collider_info)
            if num_supports > 1:
                # Ambiguous support (e.g. flat face) - keep as baseline, try perturbed directions.
                continue
            else:
                break

        # For perturbed directions, check if the support points have been found before or would form a
        # degenerate simplex. The count is skipped because an EPS perturbation doesn't change which
        # support-field grid cells are looked up, so it would return the same value as i=0.
        if i == 8:
            # If this was the last iteration, we don't check if it has been found before.
            break

        # Check if the updated simplex would be a degenerate simplex.
        if func_is_new_simplex_vertex_valid(i_b, id1, id2, mink, gjk_state, collider_info):
            break

    return obj1, obj2, local_obj1, local_obj2, id1, id2, mink


@qd.func
def count_support_driver(
    i_g, d, quat: qd.types.vector(4), dyn_info: array_class.DynInfo, collider_info: array_class.ColliderInfo
):
    """
    Count the number of possible support points in the given direction,
    using thread-local quat instead of reading from geoms_state.
    """
    geom_type = dyn_info.geoms.type[i_g]
    count = 1
    if geom_type == gs.GEOM_TYPE.BOX:
        count = support_field._func_count_supports_box(d, quat)
    elif geom_type == gs.GEOM_TYPE.MESH:
        count = support_field._func_count_supports_world(i_g, d, quat, collider_info)
    return count


@qd.func
def func_count_support(
    i_ga,
    i_gb,
    quat_a: qd.types.vector(4),
    quat_b: qd.types.vector(4),
    dir,
    dyn_info: array_class.DynInfo,
    collider_info: array_class.ColliderInfo,
):
    """
    Count the number of possible pairs of support points on the two objects
    in the given direction, using thread-local pos/quat for both geometries.
    """
    count = 1
    for i in range(2):
        count *= count_support_driver(
            i_ga if i == 0 else i_gb, dir if i == 0 else -dir, quat_a if i == 0 else quat_b, dyn_info, collider_info
        )

    return count


from genesis.utils.deprecated_module_wrapper import create_virtual_deprecated_module

create_virtual_deprecated_module(__name__, "genesis.engine.solvers.rigid.gjk_decomp")
