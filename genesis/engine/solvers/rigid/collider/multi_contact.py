"""Support-based contact manifold construction for convex geoms."""

import quadrants as qd

import genesis as gs
import genesis.utils.array_class as array_class
import genesis.utils.geom as gu

from . import mpr, support_field


@qd.func
def func_safe_normalize(v: qd.types.vector(3), collider_info: array_class.ColliderInfo):
    """
    Normalize the vector [v] safely.
    """
    norm = v.norm()

    if norm < collider_info.gjk.FLOAT_MIN[None]:
        # If the vector is too small, set it to a default value
        v[0] = 1.0
        v[1] = 0.0
        v[2] = 0.0
    else:
        # Normalize the vector
        inv_norm = 1.0 / norm
        v *= inv_norm
    return v


@qd.func
def func_support_contact_patch(
    i_ga: int,
    i_gb: int,
    i_b: int,
    normal: qd.types.vector(3),
    contact_pos: qd.types.vector(3),
    axis_0: qd.types.vector(3),
    axis_1: qd.types.vector(3),
    ga_pos: qd.types.vector(3),
    ga_quat: qd.types.vector(4),
    gb_pos: qd.types.vector(3),
    gb_quat: qd.types.vector(4),
    geoms_init_AABB: array_class.GeomsInitAABB,
    collider_state: array_class.ColliderState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    collider_info: array_class.ColliderInfo,
    rigid_config: qd.template(),
    collider_static_config: qd.template(),
    geom_pair_scale: float,
):
    """Recover contacts by clipping sampled supporting faces of two convex geoms.

    Samples retain their height along the contact normal. Coplanarity and a support query along each fitted face
    normal certify the reconstructed surface witnesses. Bounded refinement improves the overlap coverage. Return
    contact positions, individual depths and their count. An unresolved feature returns an empty patch.
    """
    EPS = rigid_info.EPS[None]
    # Support points that coincide up to the rounding of their world coordinates and of contact 0 they are taken from,
    # which grows with their distance to the origin, are one point. Distinct points lie a fraction of the geometry
    # apart, far beyond the square root of the machine precision.
    coincidence_tolerance = qd.sqrt(EPS) * geom_pair_scale + 2.0 * EPS * contact_pos.norm()

    # Two opposite supports seed a convex outline. Each query adds its support to the hull, retaining at most thirty-two
    # vertices in contact-plane coordinates plus height along the normal. Geom A lies on the positive side of the
    # normal, so its face looks backwards. Sorted geom types put a plane at A and a terrain at B.
    # Sample and query caps bound local storage and support-query count independently of mesh resolution.
    polygons = qd.Matrix.zero(gs.qd_float, 64, 3)
    polygons_size = qd.Vector.zero(gs.qd_int, 2)
    # Smooth point contacts retain the detector witness.
    plane_rounding = 8.0 * EPS * (geom_pair_scale + contact_pos.norm())
    surface_tolerance = 0.5 * collider_info.mpr.CCD_TOLERANCE[None] * geom_pair_scale
    plane_tolerance = plane_rounding + surface_tolerance
    is_line = qd.Vector.zero(gs.qd_int, 2)
    is_plane_a = dyn_info.geoms.type[i_ga] == gs.GEOM_TYPE.PLANE
    is_outlined = dyn_info.geoms.type[i_gb] != gs.GEOM_TYPE.TERRAIN
    for i_g in qd.static(range(2)):
        geom_type = dyn_info.geoms.type[i_ga if i_g == 0 else i_gb]
        if geom_type == gs.GEOM_TYPE.SPHERE or geom_type == gs.GEOM_TYPE.ELLIPSOID:
            is_outlined = False
    is_adaptive = False
    is_plane_valid = qd.Vector.zero(gs.qd_int, 2)
    is_plane_certified = qd.Vector.zero(gs.qd_int, 2)
    planes = qd.Matrix.zero(gs.qd_float, 2, 3)
    face_normals = qd.Matrix.zero(gs.qd_float, 2, 3)
    face_anchors = qd.Matrix.zero(gs.qd_float, 2, 3)
    is_mesh_plane = qd.Vector([1, 1])
    is_boundary = qd.Vector.zero(gs.qd_int, 2)
    lines_reoriented = 0
    if is_outlined:
        if is_plane_a:
            normal_local = qd.Vector(
                [dyn_info.geoms.data[i_ga][0], dyn_info.geoms.data[i_ga][1], dyn_info.geoms.data[i_ga][2]]
            )
            face_normal = gu.qd_transform_by_quat(normal_local, ga_quat)
            normal_z = face_normal.dot(normal)
            planes[0, 0] = -face_normal.dot(axis_0) / normal_z
            planes[0, 1] = -face_normal.dot(axis_1) / normal_z
            planes[0, 2] = face_normal.dot(ga_pos - contact_pos) / normal_z
            is_plane_valid[0] = True
            is_boundary[0] = True
        for i_g in range(1 if is_plane_a else 0, 2):
            geom_idx = i_ga if i_g == 0 else i_gb
            geom_pos = ga_pos if i_g == 0 else gb_pos
            geom_quat = ga_quat if i_g == 0 else gb_quat
            sign = -1.0 if i_g == 0 else 1.0
            direction_local = gu.qd_inv_transform_by_quat(sign * normal, geom_quat)
            support_local, i_v = support_field._func_support_mesh(geom_idx, direction_local, dyn_info, collider_info)
            vert_start = dyn_info.geoms.vert_start[geom_idx]
            table_start = collider_info.support_field.support_vertex_start[geom_idx]
            i_v = table_start + i_v - vert_start
            normal_local, boundary = support_field._func_support_face_normal(
                i_v, direction_local, surface_tolerance, collider_info.support_field, EPS
            )
            face_normal = gu.qd_transform_by_quat(normal_local, geom_quat)
            for i_axis in qd.static(range(3)):
                face_normals[i_g, i_axis] = normal_local[i_axis]
                face_anchors[i_g, i_axis] = support_local[i_axis]
            support_mesh = gu.qd_transform_by_trans_quat(support_local, geom_pos, geom_quat)
            support = support_mesh
            if dyn_info.geoms.type[geom_idx] != gs.GEOM_TYPE.MESH and dyn_info.geoms.type[geom_idx] != gs.GEOM_TYPE.BOX:
                # A curved native surface follows the detector normal when the mesh face fails plane validation
                for i_plane in range(2):
                    if i_plane == 0 or is_mesh_plane[i_g] == 0:
                        support = mpr.support_driver(
                            geom_idx,
                            i_b,
                            face_normal.normalized(EPS),
                            geom_pos,
                            geom_quat,
                            collider_state,
                            dyn_info,
                            collider_info,
                            rigid_config,
                            collider_static_config,
                        )
                        if i_plane == 0 and qd.abs(face_normal.dot(support - support_mesh)) > plane_tolerance:
                            is_mesh_plane[i_g] = False
                            face_normal = sign * normal
                            normal_local = gu.qd_inv_transform_by_quat(face_normal, geom_quat)
                for i_axis in qd.static(range(3)):
                    face_normals[i_g, i_axis] = normal_local[i_axis]
            normal_z = face_normal.dot(normal)
            if qd.abs(normal_z) > EPS:
                support_offset = support - contact_pos
                if (
                    dyn_info.geoms.type[geom_idx] == gs.GEOM_TYPE.MESH
                    or dyn_info.geoms.type[geom_idx] == gs.GEOM_TYPE.BOX
                ):
                    support_offset = gu.qd_transform_by_quat(support_local, geom_quat) + (geom_pos - contact_pos)
                planes[i_g, 0] = -face_normal.dot(axis_0) / normal_z
                planes[i_g, 1] = -face_normal.dot(axis_1) / normal_z
                planes[i_g, 2] = face_normal.dot(support_offset) / normal_z
                is_plane_valid[i_g] = True
                is_plane_certified[i_g] = True
                n_vertices = boundary[1] - boundary[0]
                if (
                    dyn_info.geoms.type[geom_idx] == gs.GEOM_TYPE.MESH
                    or dyn_info.geoms.type[geom_idx] == gs.GEOM_TYPE.BOX
                ):
                    if n_vertices >= 3 and n_vertices <= 8:
                        is_boundary[i_g] = True
                        for i_e in range(n_vertices):
                            i_e_ = i_e if normal_z > 0.0 else n_vertices - 1 - i_e
                            i_v_ = collider_info.support_field.support_boundary_vertex[boundary[0] + i_e_]
                            i_v = vert_start + i_v_
                            point_local = dyn_info.verts.init_pos[i_v]
                            point = gu.qd_transform_by_quat(point_local, geom_quat) + (geom_pos - contact_pos)
                            if qd.abs(normal_local.dot(point_local - support_local)) > plane_tolerance:
                                is_boundary[i_g] = False
                            polygons[32 * i_g + i_e, 0] = point.dot(axis_0)
                            polygons[32 * i_g + i_e, 1] = point.dot(axis_1)
                            polygons[32 * i_g + i_e, 2] = point.dot(normal)
                        if is_boundary[i_g]:
                            polygons_size[i_g] = n_vertices
        geoms_diameter = qd.Vector(
            [
                (geoms_init_AABB[i_ga, 7] - geoms_init_AABB[i_ga, 0]).norm(),
                (geoms_init_AABB[i_gb, 7] - geoms_init_AABB[i_gb, 0]).norm(),
            ]
        )
        # Prioritize edges nearest contact 0 so the vertex budget resolves the portion around the contact first.
        # Surviving edges retain their settled bits when an insertion removes the visible chain.
        edges_settled = qd.Vector.zero(gs.qd_int, 2)
        for i_g in range(2):
            if is_boundary[i_g]:
                edges_settled[i_g] = (1 << polygons_size[i_g]) - 1
        # Spend additional queries only where truncation leaves overlap uncertainty
        polygons_capacity = qd.Vector([8, 8])
        i_g_refine = 1 if is_plane_a else 0
        i_e_refine = 0
        n_adaptive = 0
        i_g_fit = -1
        # Four initial bounds plus the remaining support half-planes and the depth half-plane allow 65 vertices
        region_outer = qd.Matrix.zero(gs.qd_float, 65, 2)
        region_clipped = qd.Matrix.zero(gs.qd_float, 65, 2)
        n_outer = 0
        bounds = qd.Vector.zero(gs.qd_float, 4)
        sampling_tangent = 2.0 * collider_info.mc_perturbation[None]
        clip_normal_pending = qd.Vector.zero(gs.qd_float, 2)
        clip_offset_pending = gs.qd_float(0.0)
        query_offset = 4 if is_plane_a else 0
        i_g_reseed = -1
        for i_query_ in range(141):
            i_query = i_query_ + query_offset
            if i_query == 4 or i_query == 8:
                i_g_previous = i_query // 4 - 1
                if (
                    is_mesh_plane[i_g_previous] == 0
                    and polygons_size[i_g_previous] == 2
                    and lines_reoriented & (1 << i_g_previous) == 0
                ):
                    lines_reoriented |= 1 << i_g_previous
                    i_k = 32 * i_g_previous
                    line = qd.Vector.zero(gs.qd_float, 3)
                    for i_3 in qd.static(range(3)):
                        line[i_3] = polygons[i_k + 1, i_3] - polygons[i_k, i_3]
                    line_projected_sqr = line[0] ** 2 + line[1] ** 2
                    if line_projected_sqr > coincidence_tolerance**2:
                        # A line constrains its plane slope along the line alone. Minimize the remaining slope so
                        # a curved supporting feature follows the detector normal instead of the tessellated hull.
                        gradient = line[2] * qd.Vector([line[0], line[1]]) / line_projected_sqr
                        gradient_delta = gradient - qd.Vector([planes[i_g_previous, 0], planes[i_g_previous, 1]])
                        if gradient_delta.norm_sqr() > EPS**2:
                            planes[i_g_previous, 0] = gradient[0]
                            planes[i_g_previous, 1] = gradient[1]
                            polygons_size[i_g_previous] = 0
                            is_plane_certified[i_g_previous] = False
                            i_g_reseed = i_g_previous
                            query_offset -= 4
                            i_query -= 4
            for _ in range(2):
                if i_query < 8:
                    if is_boundary[i_query // 4]:
                        query_offset += 4
                        i_query += 4
            if i_g_fit >= 0:
                is_plane_certified[i_g_fit] = False
                n_points = polygons_size[i_g_fit]
                plane_normal = qd.Vector.zero(gs.qd_float, 3)
                centroid = qd.Vector.zero(gs.qd_float, 3)
                for j_v in range(n_points):
                    j_k0 = 32 * i_g_fit + j_v
                    j_k1 = 32 * i_g_fit + (j_v + 1) % n_points
                    plane_normal[0] += (polygons[j_k0, 1] - polygons[j_k1, 1]) * (polygons[j_k0, 2] + polygons[j_k1, 2])
                    plane_normal[1] += (polygons[j_k0, 2] - polygons[j_k1, 2]) * (polygons[j_k0, 0] + polygons[j_k1, 0])
                    plane_normal[2] += (polygons[j_k0, 0] - polygons[j_k1, 0]) * (polygons[j_k0, 1] + polygons[j_k1, 1])
                    for i_3 in qd.static(range(3)):
                        centroid[i_3] += polygons[j_k0, i_3] / n_points
                gradient = qd.Vector.zero(gs.qd_float, 2)
                if qd.abs(plane_normal[2]) > EPS * plane_normal.norm():
                    gradient = -qd.Vector([plane_normal[0], plane_normal[1]]) / plane_normal[2]
                height_origin = centroid[2] - gradient[0] * centroid[0] - gradient[1] * centroid[1]
                planes[i_g_fit, 0], planes[i_g_fit, 1], planes[i_g_fit, 2] = gradient[0], gradient[1], height_origin
                is_plane_valid[i_g_fit] = n_points >= 3 and qd.abs(plane_normal[2]) > EPS * geom_pair_scale**2
                for j_v in range(n_points):
                    j_k = 32 * i_g_fit + j_v
                    height = height_origin + gradient[0] * polygons[j_k, 0] + gradient[1] * polygons[j_k, 1]
                    if qd.abs(polygons[j_k, 2] - height) > plane_tolerance:
                        is_plane_valid[i_g_fit] = False
                i_g_fit = -1
            # Finish the last plane update without spending another support query.
            if i_query == 132:
                break
            i_g = i_query // 4
            i_e = 0
            direction_x = gs.qd_float(0.0)
            direction_y = gs.qd_float(0.0)
            if i_query < 8:
                seed_quat = ga_quat if i_g == 0 else gb_quat
                normal_local = gu.qd_inv_transform_by_quat(normal, seed_quat)
                seed_0, seed_1 = gu.qd_orthogonals(normal_local)
                seed_local = seed_0 if i_query % 2 == 0 else seed_1
                seed_world = gu.qd_transform_by_quat(seed_local, seed_quat)
                seed_sign = 1.0 if i_query % 4 < 2 else -1.0
                direction_x = seed_sign * seed_world.dot(axis_0)
                direction_y = seed_sign * seed_world.dot(axis_1)
            is_query = i_query < 8
            if not is_query and not is_adaptive:
                for _ in range(2):
                    if i_g_refine < 2:
                        n_points = polygons_size[i_g_refine]
                        if (
                            n_points < 2
                            or n_points >= polygons_capacity[i_g_refine]
                            or edges_settled[i_g_refine] == (1 << n_points) - 1
                        ):
                            i_g_refine += 1
                            i_e_refine = 0
                if i_g_refine < 2:
                    i_e_refine = 0
                    distance_min = gs.qd_float(-1.0)
                    for j_e in range(polygons_size[i_g_refine]):
                        if edges_settled[i_g_refine] & (1 << j_e) == 0:
                            j_k0 = 32 * i_g_refine + j_e
                            j_k1 = 32 * i_g_refine + (j_e + 1) % polygons_size[i_g_refine]
                            edge_x = polygons[j_k1, 0] - polygons[j_k0, 0]
                            edge_y = polygons[j_k1, 1] - polygons[j_k0, 1]
                            edge_length_sqr = edge_x**2 + edge_y**2
                            edge_fraction = -(polygons[j_k0, 0] * edge_x + polygons[j_k0, 1] * edge_y)
                            edge_fraction = qd.min(qd.max(edge_fraction / qd.max(edge_length_sqr, EPS**2), 0.0), 1.0)
                            distance = (polygons[j_k0, 0] + edge_fraction * edge_x) ** 2
                            distance += (polygons[j_k0, 1] + edge_fraction * edge_y) ** 2
                            if distance_min < 0.0 or distance < distance_min:
                                distance_min = distance
                                i_e_refine = j_e
                    # Both directed seed edges span the same segment. Contact 0 selects the side to refine first
                    if polygons_size[i_g_refine] == 2 and edges_settled[i_g_refine] == 0:
                        i_k0 = 32 * i_g_refine
                        edge_x = polygons[i_k0 + 1, 0] - polygons[i_k0, 0]
                        edge_y = polygons[i_k0 + 1, 1] - polygons[i_k0, 1]
                        side = edge_y * polygons[i_k0, 0] - edge_x * polygons[i_k0, 1]
                        i_e_refine = 0 if side <= 0.0 else 1
                    i_g, i_e = i_g_refine, i_e_refine
                    n_points = polygons_size[i_g]
                    i_k0 = 32 * i_g + i_e
                    i_k1 = 32 * i_g + (i_e + 1) % n_points
                    edge_x = polygons[i_k1, 0] - polygons[i_k0, 0]
                    edge_y = polygons[i_k1, 1] - polygons[i_k0, 1]
                    edge_norm = qd.sqrt(edge_x**2 + edge_y**2)
                    direction_x, direction_y = edge_y / edge_norm, -edge_x / edge_norm
                    is_query = True
            if not is_query and not is_adaptive:
                # Refine truncated planar outlines until the uncertainty is small in the overlap itself
                is_adaptive = not is_plane_a and not (is_line != 0).any()
                is_adaptive = is_adaptive and not (is_boundary[0] and is_boundary[1])
                is_adaptive = is_adaptive and (polygons_size[0] >= 8 or polygons_size[1] >= 8)
                is_adaptive = is_adaptive and is_plane_valid[0] != 0 and is_plane_valid[1] != 0
            if is_adaptive:
                i_g = 0
                is_query = n_adaptive < 64
                n_clip_outer = 0
                if n_adaptive > 4:
                    n_clip_outer = 2 if n_adaptive == 8 else 1
                for j_clip in range(n_clip_outer):
                    clip_normal = qd.Vector([planes[0, 0] - planes[1, 0], planes[0, 1] - planes[1, 1]])
                    clip_offset = planes[1, 2] - planes[0, 2]
                    if j_clip == 0:
                        clip_normal = clip_normal_pending
                        clip_offset = clip_offset_pending
                    n_clipped = 0
                    for j_v in range(n_outer):
                        j_v_next = (j_v + 1) % n_outer
                        point = qd.Vector([region_outer[j_v, 0], region_outer[j_v, 1]])
                        point_next = qd.Vector([region_outer[j_v_next, 0], region_outer[j_v_next, 1]])
                        side = clip_offset - clip_normal.dot(point)
                        side_next = clip_offset - clip_normal.dot(point_next)
                        if side >= 0.0:
                            region_clipped[n_clipped, 0], region_clipped[n_clipped, 1] = point[0], point[1]
                            n_clipped += 1
                        if (side >= 0.0) != (side_next >= 0.0):
                            fraction = side / (side - side_next)
                            point += fraction * (point_next - point)
                            region_clipped[n_clipped, 0], region_clipped[n_clipped, 1] = point[0], point[1]
                            n_clipped += 1
                    for j_v in range(n_clipped):
                        region_outer[j_v, 0], region_outer[j_v, 1] = region_clipped[j_v, 0], region_clipped[j_v, 1]
                    n_outer = n_clipped
                if n_adaptive < 8:
                    i_g = n_adaptive // 4
                    i_d = n_adaptive % 4
                    direction_x = 1.0 if i_d == 0 else (-1.0 if i_d == 2 else 0.0)
                    direction_y = 1.0 if i_d == 1 else (-1.0 if i_d == 3 else 0.0)
                else:
                    area_outer = gs.qd_float(0.0)
                    perimeter_outer = gs.qd_float(0.0)
                    bounds_min = qd.Vector([region_outer[0, 0], region_outer[0, 1]])
                    bounds_max = bounds_min
                    distance_max = gs.qd_float(-1.0)
                    for j_v in range(n_outer):
                        point = qd.Vector([region_outer[j_v, 0], region_outer[j_v, 1]])
                        j_v1 = (j_v + 1) % n_outer
                        area_outer += point[0] * region_outer[j_v1, 1] - point[1] * region_outer[j_v1, 0]
                        edge_outer = qd.Vector([region_outer[j_v1, 0], region_outer[j_v1, 1]]) - point
                        perimeter_outer += edge_outer.norm()
                        bounds_min = qd.min(bounds_min, point)
                        bounds_max = qd.max(bounds_max, point)
                        for j_g in range(2):
                            n_points = polygons_size[j_g]
                            distance_min = gs.qd_float(1e30)
                            direction_nearest = qd.Vector.zero(gs.qd_float, 2)
                            violation_max = gs.qd_float(-1.0)
                            is_inside = n_points >= 3
                            for j_e in range(n_points):
                                j_e1 = (j_e + 1) % n_points
                                j_k0, j_k1 = 32 * j_g + j_e, 32 * j_g + j_e1
                                point_start = qd.Vector([polygons[j_k0, 0], polygons[j_k0, 1]])
                                point_end = qd.Vector([polygons[j_k1, 0], polygons[j_k1, 1]])
                                edge = point_end - point_start
                                delta = point - point_start
                                side = edge[0] * delta[1] - edge[1] * delta[0]
                                edge_length_sqr = qd.max(edge.dot(edge), EPS**2)
                                if side < 0.0:
                                    is_inside = False
                                    violation = side**2 / edge_length_sqr
                                    if violation > violation_max:
                                        violation_max = violation
                                        direction_nearest = qd.Vector([edge[1], -edge[0]])
                                fraction = qd.min(qd.max(delta.dot(edge) / edge_length_sqr, 0.0), 1.0)
                                delta -= fraction * edge
                                if delta.dot(delta) < distance_min:
                                    distance_min = delta.dot(delta)
                            if is_inside:
                                distance_min = 0.0
                            if distance_min > distance_max:
                                distance_max = distance_min
                                i_g = j_g
                                # Edge normals keep a query fixed while the overlap moves past the same sampled edge
                                direction_x, direction_y = direction_nearest[0], direction_nearest[1]
                    diameter_sqr = (bounds_max - bounds_min).norm_sqr()
                    # Eroding the outer overlap by its maximum distance to either sampled face lies inside both
                    # faces. The removed area is at most that distance times the outer perimeter.
                    if n_outer < 3 or (
                        2.0 * qd.sqrt(qd.max(distance_max, 0.0)) * perimeter_outer <= 0.02 * area_outer
                        and distance_max <= 0.005**2 * diameter_sqr
                    ):
                        is_query = False
                    if distance_max <= coincidence_tolerance**2 or polygons_size[i_g] >= 32:
                        is_query = False
                    direction_norm = qd.sqrt(direction_x**2 + direction_y**2)
                    direction_x /= qd.max(direction_norm, EPS)
                    direction_y /= qd.max(direction_norm, EPS)
            if not is_query:
                break
            if is_query:
                is_b = i_g == 1
                edge_tolerance = coincidence_tolerance
                if is_plane_valid[i_g] and (not is_adaptive or n_adaptive >= 8):
                    # A one-sided query tilt resolves tied supports. Its support error is bounded by the tilt times
                    # the geom diameter, sharing the existing edge tolerance with the insertion test.
                    direction_bias = 0.5 * coincidence_tolerance / qd.max(geoms_diameter[i_g], EPS)
                    direction_biased = qd.Vector(
                        [
                            direction_x - direction_bias * direction_y,
                            direction_y + direction_bias * direction_x,
                        ]
                    ).normalized(EPS)
                    direction_x, direction_y = direction_biased[0], direction_biased[1]
                    edge_tolerance *= 0.5
                tangent = sampling_tangent
                if is_mesh_plane[i_g] == 0:
                    # Resolve supports throughout the detector's surface band when the native normal is approximate.
                    tangent = qd.max(0.125 * coincidence_tolerance, surface_tolerance) / qd.max(
                        geoms_diameter[i_g], EPS
                    )
                direction = tangent * (direction_x * axis_0 + direction_y * axis_1) + (normal if is_b else -normal)
                sign = 1.0 if is_b else -1.0
                if is_plane_valid[i_g]:
                    # Removing the face slope makes the support certify an edge in the projected outline.
                    direction -= sign * (planes[i_g, 0] * axis_0 + planes[i_g, 1] * axis_1)
                geom_idx = i_gb if is_b else i_ga
                geom_pos = gb_pos if is_b else ga_pos
                geom_quat = gb_quat if is_b else ga_quat
                support_offset = qd.Vector.zero(gs.qd_float, 3)
                if (
                    dyn_info.geoms.type[geom_idx] == gs.GEOM_TYPE.MESH
                    or dyn_info.geoms.type[geom_idx] == gs.GEOM_TYPE.BOX
                ):
                    normal_local = qd.Vector([face_normals[i_g, 0], face_normals[i_g, 1], face_normals[i_g, 2]])
                    anchor_local = qd.Vector([face_anchors[i_g, 0], face_anchors[i_g, 1], face_anchors[i_g, 2]])
                    tangent_local = gu.qd_inv_transform_by_quat(direction_x * axis_0 + direction_y * axis_1, geom_quat)
                    support_local = support_field._func_support_mesh_face(
                        geom_idx,
                        normal_local,
                        anchor_local,
                        tangent_local,
                        collider_info,
                        surface_tolerance * qd.abs(normal_local.dot(gu.qd_inv_transform_by_quat(normal, geom_quat))),
                        EPS,
                    )
                    support_offset = gu.qd_transform_by_quat(support_local, geom_quat) + (geom_pos - contact_pos)
                else:
                    support_offset = (
                        mpr.support_driver(
                            i_gb if is_b else i_ga,
                            i_b,
                            direction.normalized(EPS),
                            gb_pos if is_b else ga_pos,
                            gb_quat if is_b else ga_quat,
                            collider_state,
                            dyn_info,
                            collider_info,
                            rigid_config,
                            collider_static_config,
                        )
                        - contact_pos
                    )
                corner = qd.Vector([support_offset.dot(axis_0), support_offset.dot(axis_1), support_offset.dot(normal)])
                if i_g_reseed == i_g:
                    planes[i_g, 2] = corner[2] - planes[i_g, 0] * corner[0] - planes[i_g, 1] * corner[1]
                    i_g_reseed = -1
                if is_adaptive:
                    height = planes[i_g, 0] * corner[0] + planes[i_g, 1] * corner[1] + planes[i_g, 2]
                    if qd.abs(corner[2] - height) > plane_tolerance:
                        is_adaptive = False
                        break
                    offset = direction_x * corner[0] + direction_y * corner[1]
                    offset += sign * (corner[2] - height) / sampling_tangent
                    if n_adaptive < 4:
                        bounds[n_adaptive] = offset
                        if n_adaptive == 3:
                            region_outer[0, 0], region_outer[0, 1] = -bounds[2], -bounds[3]
                            region_outer[1, 0], region_outer[1, 1] = bounds[0], -bounds[3]
                            region_outer[2, 0], region_outer[2, 1] = bounds[0], bounds[1]
                            region_outer[3, 0], region_outer[3, 1] = -bounds[2], bounds[1]
                            n_outer = 4
                    else:
                        clip_normal_pending = qd.Vector([direction_x, direction_y])
                        clip_offset_pending = offset
                    n_adaptive += 1
                    side_min = gs.qd_float(0.0)
                    i_e = 0
                    for j_e in range(polygons_size[i_g]):
                        j_k0 = 32 * i_g + j_e
                        j_k1 = 32 * i_g + (j_e + 1) % polygons_size[i_g]
                        edge_x = polygons[j_k1, 0] - polygons[j_k0, 0]
                        edge_y = polygons[j_k1, 1] - polygons[j_k0, 1]
                        side = edge_x * (corner[1] - polygons[j_k0, 1]) - edge_y * (corner[0] - polygons[j_k0, 0])
                        if side < side_min:
                            side_min = side
                            i_e = j_e
                n_points = polygons_size[i_g]
                if i_query < 8:
                    is_distinct = True
                    for i_v in range(n_points):
                        i_k = 32 * i_g + i_v
                        gap_x, gap_y = corner[0] - polygons[i_k, 0], corner[1] - polygons[i_k, 1]
                        if gap_x**2 + gap_y**2 <= coincidence_tolerance**2:
                            is_distinct = False
                    if is_distinct:
                        for i_3 in qd.static(range(3)):
                            polygons[32 * i_g + n_points, i_3] = corner[i_3]
                        polygons_size[i_g] = n_points + 1
                elif not is_adaptive or n_adaptive > 8:
                    # Outer bounds certify the overlap. Edge-normal queries alone refine its sampled outline.
                    i_k0 = 32 * i_g + i_e
                    i_k1 = 32 * i_g + (i_e + 1) % n_points
                    edge_x, edge_y = polygons[i_k1, 0] - polygons[i_k0, 0], polygons[i_k1, 1] - polygons[i_k0, 1]
                    side = edge_x * (corner[1] - polygons[i_k0, 1]) - edge_y * (corner[0] - polygons[i_k0, 0])
                    if side < -edge_tolerance * qd.sqrt(edge_x**2 + edge_y**2):
                        i_g_fit = i_g
                        is_local = True
                        for j_e_ in range(2):
                            j_e = (i_e + n_points - 1 + 2 * j_e_) % n_points
                            j_k0 = 32 * i_g + j_e
                            j_k1 = 32 * i_g + (j_e + 1) % n_points
                            neighbor_x = polygons[j_k1, 0] - polygons[j_k0, 0]
                            neighbor_y = polygons[j_k1, 1] - polygons[j_k0, 1]
                            neighbor_side = neighbor_x * (corner[1] - polygons[j_k0, 1])
                            neighbor_side -= neighbor_y * (corner[0] - polygons[j_k0, 0])
                            if neighbor_side < 0.0:
                                is_local = False
                        if is_local:
                            for i_v_ in range(n_points - i_e - 1):
                                i_v = n_points - i_v_
                                for i_3 in qd.static(range(3)):
                                    polygons[32 * i_g + i_v, i_3] = polygons[32 * i_g + i_v - 1, i_3]
                            for i_3 in qd.static(range(3)):
                                polygons[32 * i_g + i_e + 1, i_3] = corner[i_3]
                            polygons_size[i_g] = n_points + 1
                            edges_lower = edges_settled[i_g] & ((1 << i_e) - 1)
                            edges_upper = (edges_settled[i_g] >> (i_e + 1)) << (i_e + 2)
                            edges_settled[i_g] = edges_lower | edges_upper
                        else:
                            edges_visible = gs.qd_int(0)
                            for j_e in range(n_points):
                                j_k0 = 32 * i_g + j_e
                                j_k1 = 32 * i_g + (j_e + 1) % n_points
                                edge_x = polygons[j_k1, 0] - polygons[j_k0, 0]
                                edge_y = polygons[j_k1, 1] - polygons[j_k0, 1]
                                side = edge_x * (corner[1] - polygons[j_k0, 1])
                                side -= edge_y * (corner[0] - polygons[j_k0, 0])
                                if side < 0.0:
                                    edges_visible |= 1 << j_e
                            i_v_start = 0
                            i_v_end = 0
                            n_visible_runs = 0
                            for j_e in range(n_points):
                                is_visible = edges_visible & (1 << j_e) != 0
                                is_prev_visible = edges_visible & (1 << ((j_e + n_points - 1) % n_points)) != 0
                                if is_prev_visible and not is_visible:
                                    i_v_start = j_e
                                if is_visible and not is_prev_visible:
                                    i_v_end = j_e
                                    n_visible_runs += 1
                            if n_visible_runs == 1:
                                n_kept = (i_v_end + n_points - i_v_start) % n_points + 1
                                edges_settled_next = (edges_settled[i_g] >> i_v_start) | (
                                    edges_settled[i_g] << (n_points - i_v_start)
                                )
                                edges_settled_next &= (1 << (n_kept - 1)) - 1
                                if i_v_start > 0:
                                    # The kept chain starts at vertex 0 after a rotation by i_v_start, made of as many
                                    # cycles as the greatest common divisor of both counts. Euclid's algorithm takes at
                                    # most 7 steps for counts up to 32.
                                    n_cycles = n_points
                                    n_remainder = i_v_start
                                    for _ in range(7):
                                        if n_remainder > 0:
                                            n_next = n_cycles % n_remainder
                                            n_cycles = n_remainder
                                            n_remainder = n_next
                                    for j_v in range(n_cycles):
                                        i_k = 32 * i_g + j_v
                                        corner_cycle = qd.Vector([polygons[i_k, 0], polygons[i_k, 1], polygons[i_k, 2]])
                                        i_v = j_v
                                        for _ in range(n_points // n_cycles - 1):
                                            i_v_next = (i_v + i_v_start) % n_points
                                            for i_3 in qd.static(range(3)):
                                                polygons[32 * i_g + i_v, i_3] = polygons[32 * i_g + i_v_next, i_3]
                                            i_v = i_v_next
                                        for i_3 in qd.static(range(3)):
                                            polygons[32 * i_g + i_v, i_3] = corner_cycle[i_3]
                                for i_3 in qd.static(range(3)):
                                    polygons[32 * i_g + n_kept, i_3] = corner[i_3]
                                polygons_size[i_g] = n_kept + 1
                                edges_settled[i_g] = edges_settled_next
                            else:
                                # Roundoff can split nearly collinear visible edges into several chains. Rebuild the
                                # hull to restore convex vertex order before clipping.
                                for i_3 in qd.static(range(3)):
                                    polygons[32 * i_g + n_points, i_3] = corner[i_3]
                                n_points += 1
                                i_k_start = 0
                                for j_k_ in range(1, n_points):
                                    i_k = 32 * i_g + i_k_start
                                    j_k = 32 * i_g + j_k_
                                    is_before = polygons[j_k, 0] < polygons[i_k, 0]
                                    is_before = is_before or (
                                        polygons[j_k, 0] <= polygons[i_k, 0] and polygons[j_k, 1] < polygons[i_k, 1]
                                    )
                                    if is_before:
                                        i_k_start = j_k_
                                for i_3 in qd.static(range(3)):
                                    coordinate = polygons[32 * i_g, i_3]
                                    polygons[32 * i_g, i_3] = polygons[32 * i_g + i_k_start, i_3]
                                    polygons[32 * i_g + i_k_start, i_3] = coordinate
                                n_hull = 0
                                is_closed = False
                                for i_v in range(n_points):
                                    if not is_closed:
                                        n_hull = i_v + 1
                                        i_k0 = 32 * i_g + i_v
                                        i_k_next = 1 if i_v == 0 and n_points > 1 else 0
                                        for j_k_ in range(i_v + 1, n_points):
                                            i_k1 = 32 * i_g + i_k_next
                                            j_k = 32 * i_g + j_k_
                                            edge_x = polygons[i_k1, 0] - polygons[i_k0, 0]
                                            edge_y = polygons[i_k1, 1] - polygons[i_k0, 1]
                                            delta_x = polygons[j_k, 0] - polygons[i_k0, 0]
                                            delta_y = polygons[j_k, 1] - polygons[i_k0, 1]
                                            side = edge_x * delta_y - edge_y * delta_x
                                            distance = delta_x**2 + delta_y**2
                                            edge_length_sqr = edge_x**2 + edge_y**2
                                            side_tolerance = EPS * qd.sqrt(edge_length_sqr * distance)
                                            is_farther = qd.abs(side) <= side_tolerance and distance > edge_length_sqr
                                            if side < -side_tolerance or is_farther:
                                                i_k_next = j_k_
                                        is_closed = i_k_next == 0
                                        if not is_closed:
                                            for i_3 in qd.static(range(3)):
                                                coordinate = polygons[32 * i_g + n_hull, i_3]
                                                polygons[32 * i_g + n_hull, i_3] = polygons[32 * i_g + i_k_next, i_3]
                                                polygons[32 * i_g + i_k_next, i_3] = coordinate
                                polygons_size[i_g] = n_hull
                                edges_settled[i_g] = 0
                    else:
                        edges_settled[i_g] |= 1 << i_e
                if i_g_fit >= 0 and is_plane_valid[i_g] != 0:
                    # Retained vertices already satisfy the plane, so only the inserted support needs checking.
                    height = planes[i_g, 2] + planes[i_g, 0] * corner[0] + planes[i_g, 1] * corner[1]
                    if qd.abs(corner[2] - height) <= plane_tolerance:
                        i_g_fit = -1

    if is_outlined:
        for i_g in range(2):
            is_line[i_g] = polygons_size[i_g] == 2
        if is_plane_a:
            # An unbounded plane covers the other feature's complete projected outline.
            polygons_size[0] = polygons_size[1]
            for i_v in range(polygons_size[1]):
                polygons[i_v, 0], polygons[i_v, 1] = polygons[32 + i_v, 0], polygons[32 + i_v, 1]
        is_outlined = is_plane_valid[0] != 0 and is_plane_valid[1] != 0
        for i_g in range(1 if is_plane_a else 0, 2):
            if is_outlined:
                if is_mesh_plane[i_g] == 0 and polygons_size[i_g] == 2 and lines_reoriented & (1 << i_g) != 0:
                    i_k = 32 * i_g
                    line = qd.Vector.zero(gs.qd_float, 3)
                    for i_3 in qd.static(range(3)):
                        line[i_3] = polygons[i_k + 1, i_3] - polygons[i_k, i_3]
                    line_projected_sqr = line[0] ** 2 + line[1] ** 2
                    if line_projected_sqr > coincidence_tolerance**2:
                        gradient = line[2] * qd.Vector([line[0], line[1]]) / line_projected_sqr
                        planes[i_g, 0], planes[i_g, 1] = gradient[0], gradient[1]
                        planes[i_g, 2] = polygons[i_k, 2] - gradient[0] * polygons[i_k, 0]
                        planes[i_g, 2] -= gradient[1] * polygons[i_k, 1]
                if is_plane_certified[i_g]:
                    # The supporting plane is exact, but tilted seed queries can reach an adjacent face.
                    for i_v in range(polygons_size[i_g]):
                        i_k = 32 * i_g + i_v
                        height = planes[i_g, 2] + planes[i_g, 0] * polygons[i_k, 0]
                        height += planes[i_g, 1] * polygons[i_k, 1]
                        if qd.abs(polygons[i_k, 2] - height) > plane_tolerance:
                            is_outlined = False
                else:
                    sign = -1.0 if i_g == 0 else 1.0
                    direction = sign * (normal - planes[i_g, 0] * axis_0 - planes[i_g, 1] * axis_1)
                    support = mpr.support_driver(
                        i_ga if i_g == 0 else i_gb,
                        i_b,
                        direction.normalized(EPS),
                        ga_pos if i_g == 0 else gb_pos,
                        ga_quat if i_g == 0 else gb_quat,
                        collider_state,
                        dyn_info,
                        collider_info,
                        rigid_config,
                        collider_static_config,
                    )
                    # Coplanar samples may span an interior section. The extremal height certifies a boundary plane.
                    distance = direction.dot(support - contact_pos) - sign * planes[i_g, 2]
                    if qd.abs(distance) > plane_tolerance * direction.norm():
                        is_outlined = False

    # Two thirty-two-vertex hulls and the depth half-plane bound at most sixty-five overlap vertices
    overlap = qd.Matrix.zero(gs.qd_float, 65, 2)
    n_overlap = 0
    slope = qd.Vector.zero(gs.qd_float, 2)
    if is_outlined:
        # The fitted face heights bound the portion of the overlap that interpenetrates.
        slope = qd.Vector([planes[1, 0] - planes[0, 0], planes[1, 1] - planes[0, 1]])
        depth_origin = planes[1, 2] - planes[0, 2]

        # Clip a segment as the subject whenever a face is available. An unbounded plane contributes only depth.
        i_subject = 1 if is_line[1] != 0 and not is_plane_a else 0
        i_clip = 1 - i_subject
        clip_start = 32 * i_clip
        n_overlap = polygons_size[i_subject]
        for i_k in range(n_overlap):
            overlap[i_k, 0] = polygons[32 * i_subject + i_k, 0]
            overlap[i_k, 1] = polygons[32 * i_subject + i_k, 1]
        n_clip = 0 if is_plane_a else polygons_size[i_clip]
        if n_clip < 2 and not is_plane_a:
            n_overlap = 0
        # The overlap lies within the subject polygon, which clipping only cuts, and within the convex hull of the clip
        # polygon, which holds the intersection of the half-planes of its edges. It is thus empty when the depth, affine
        # in the plane, is negative at every corner of either polygon, which spares the clipping of the faces apart from
        # each other.
        is_pressed_subject = False
        for i_k in range(n_overlap):
            if depth_origin + slope[0] * overlap[i_k, 0] + slope[1] * overlap[i_k, 1] >= 0.0:
                is_pressed_subject = True
        is_pressed_clip = is_plane_a
        for i_e in range(n_clip):
            if (
                depth_origin + slope[0] * polygons[clip_start + i_e, 0] + slope[1] * polygons[clip_start + i_e, 1]
                >= 0.0
            ):
                is_pressed_clip = True
        if not (is_pressed_subject and is_pressed_clip):
            n_overlap = 0
            n_clip = 0
        clip_area = gs.qd_float(0.0)
        for i_e in range(n_clip):
            i_e_next = (i_e + 1) % n_clip
            clip_area += (
                polygons[clip_start + i_e, 0] * polygons[clip_start + i_e_next, 1]
                - polygons[clip_start + i_e, 1] * polygons[clip_start + i_e_next, 0]
            )
        orientation = 1.0 if clip_area >= 0.0 else -1.0
        n_halfplanes = 4 if n_clip == 2 else n_clip
        overlap_out = qd.Matrix.zero(gs.qd_float, 65, 2)
        for i_e in range(n_halfplanes + 1):
            # Each clip keeps the side where the affine function 'offset + inward_normal . p' is non-negative
            inward_normal = slope
            offset = depth_origin
            if i_e < n_halfplanes:
                i_corner = 0 if n_clip == 2 else i_e
                corner = qd.Vector([polygons[clip_start + i_corner, 0], polygons[clip_start + i_corner, 1]])
                i_next = clip_start + (i_corner + 1) % n_clip
                edge = qd.Vector([polygons[i_next, 0], polygons[i_next, 1]]) - corner
                inward_normal = orientation * qd.Vector([-edge[1], edge[0]])
                if n_clip == 2:
                    # End planes bound the segment. Its two side planes admit coordinate roundoff.
                    if i_e < 2:
                        inward_normal = (1.0 if i_e == 0 else -1.0) * edge
                        if i_e == 1:
                            corner += edge
                    elif i_e == 3:
                        inward_normal = -inward_normal
                offset = -inward_normal.dot(corner)
                if n_clip == 2 and i_e >= 2:
                    offset += plane_rounding * edge.norm()
            n_out = 0
            for i_k in range(n_overlap):
                p_x, p_y = overlap[i_k, 0], overlap[i_k, 1]
                q_x, q_y = overlap[(i_k + 1) % n_overlap, 0], overlap[(i_k + 1) % n_overlap, 1]
                side_p = offset + inward_normal[0] * p_x + inward_normal[1] * p_y
                side_q = offset + inward_normal[0] * q_x + inward_normal[1] * q_y
                if side_p >= 0.0 and n_out < 65:
                    overlap_out[n_out, 0], overlap_out[n_out, 1] = p_x, p_y
                    n_out += 1
                if side_p * side_q < 0.0 and n_out < 65:
                    crossing_fraction = side_p / (side_p - side_q)
                    overlap_out[n_out, 0] = p_x + crossing_fraction * (q_x - p_x)
                    overlap_out[n_out, 1] = p_y + crossing_fraction * (q_y - p_y)
                    n_out += 1
            for i_k in range(n_out):
                overlap[i_k, 0], overlap[i_k, 1] = overlap_out[i_k, 0], overlap_out[i_k, 1]
            n_overlap = n_out

        # Where an edge of one polygon crosses a corner of the other, clipping places a point of each polygon at that
        # corner, and both points are merged into one corner
        n_distinct = 0
        for i_k in range(n_overlap):
            is_distinct = n_distinct == 0
            if n_distinct > 0:
                gap = qd.Vector(
                    [
                        overlap[i_k, 0] - overlap[n_distinct - 1, 0],
                        overlap[i_k, 1] - overlap[n_distinct - 1, 1],
                    ]
                )
                is_distinct = gap.norm_sqr() > coincidence_tolerance**2
            if is_distinct:
                overlap[n_distinct, 0], overlap[n_distinct, 1] = overlap[i_k, 0], overlap[i_k, 1]
                n_distinct += 1
        if n_distinct > 1:
            gap = qd.Vector([overlap[n_distinct - 1, 0] - overlap[0, 0], overlap[n_distinct - 1, 1] - overlap[0, 1]])
            if gap.norm_sqr() <= coincidence_tolerance**2:
                n_distinct -= 1
        n_overlap = n_distinct

    # Removing the smallest corner triangle keeps the remaining polygon inside the certified overlap.
    axis_0_local = gu.qd_inv_transform_by_quat(axis_0, ga_quat)
    axis_1_local = gu.qd_inv_transform_by_quat(axis_1, ga_quat)
    normal_local = gu.qd_inv_transform_by_quat(normal, ga_quat)
    coordinate_tolerance = 8.0 * EPS * geom_pair_scale
    for _ in range(65):
        if n_overlap <= collider_static_config.n_contacts_per_convex_pair:
            break
        area_min = gs.qd_float(1e30)
        area_error_min = gs.qd_float(0.0)
        i_k_remove = 0
        point_remove = qd.Vector.zero(gs.qd_float, 3)
        for i_k in range(n_overlap):
            i_k_prev = (i_k + n_overlap - 1) % n_overlap
            i_k_next = (i_k + 1) % n_overlap
            edge_0 = qd.Vector([overlap[i_k, 0] - overlap[i_k_prev, 0], overlap[i_k, 1] - overlap[i_k_prev, 1]])
            edge_1 = qd.Vector([overlap[i_k_next, 0] - overlap[i_k, 0], overlap[i_k_next, 1] - overlap[i_k, 1]])
            area_lhs = edge_0[0] * edge_1[1]
            area_rhs = edge_0[1] * edge_1[0]
            area = qd.abs(area_lhs - area_rhs)
            area_error = 8.0 * EPS * (qd.abs(area_lhs) + qd.abs(area_rhs))
            # Corner-coordinate uncertainty contributes an edge-length term to the area error.
            area_error += 2.0 * coordinate_tolerance * (edge_0.norm() + edge_1.norm())
            area_error += 4.0 * coordinate_tolerance**2
            area_tolerance = area_error + area_error_min
            x, y = overlap[i_k, 0], overlap[i_k, 1]
            height = planes[0, 0] * x + planes[0, 1] * y
            point = x * axis_0_local + y * axis_1_local + height * normal_local
            # Geom-frame coordinates resolve area ties within arithmetic precision consistently.
            is_coordinate_tied = True
            is_preferred = False
            for i_3 in qd.static(range(3)):
                if is_coordinate_tied and qd.abs(point[i_3] - point_remove[i_3]) > coordinate_tolerance:
                    is_preferred = point[i_3] < point_remove[i_3]
                    is_coordinate_tied = False
            if area < area_min - area_tolerance or (qd.abs(area - area_min) <= area_tolerance and is_preferred):
                area_min = area
                area_error_min = area_error
                i_k_remove = i_k
                point_remove = point
        for i_k in range(i_k_remove, n_overlap - 1):
            overlap[i_k, 0], overlap[i_k, 1] = overlap[i_k + 1, 0], overlap[i_k + 1, 1]
        n_overlap -= 1

    positions = qd.Matrix.zero(gs.qd_float, collider_static_config.n_contacts_per_convex_pair, 3)
    depths = qd.Vector.zero(gs.qd_float, collider_static_config.n_contacts_per_convex_pair)
    for i_c in range(n_overlap):
        x, y = overlap[i_c, 0], overlap[i_c, 1]
        height_a = planes[0, 0] * x + planes[0, 1] * y + planes[0, 2]
        height_b = planes[1, 0] * x + planes[1, 1] * y + planes[1, 2]
        position = contact_pos + x * axis_0 + y * axis_1 + 0.5 * (height_a + height_b) * normal
        for i_3 in qd.static(range(3)):
            positions[i_c, i_3] = position[i_3]
        depths[i_c] = qd.max(height_b - height_a, 0.0)
    return positions, depths, n_overlap
