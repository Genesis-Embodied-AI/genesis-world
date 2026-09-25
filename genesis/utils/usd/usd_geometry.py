import re
from typing import Dict, List

import numpy as np
import trimesh
from pxr import Usd, UsdGeom, UsdPhysics

import genesis as gs
from genesis.utils import geom as gu

from .usd_context import UsdContext
from .usd_utils import AXES_T, AXES_VECTOR, usd_attr_array_to_numpy, usd_primvar_array_to_numpy


# UsdPhysics.MeshCollisionAPI 'approximation' tokens -> per-geom collision post-processing overrides
# consumed by KinematicEntityDescription._resolve_geoms. 'boundingCube'/'boundingSphere' are handled
# separately by fitting a primitive geom in the parser.
_APPROXIMATION_OVERRIDES = {
    "convexHull": {"convexify": True, "decompose_error_threshold": float("inf")},  # single hull, no decomposition
    "convexDecomposition": {"convexify": True},  # allow decomposition per the morph threshold
    "none": {"convexify": False, "decimate": False},  # exact raw triangle mesh
    "meshSimplification": {"convexify": False, "decimate": True},  # decimated triangle mesh
    "sdf": {"convexify": False},  # signed distance field (SDF): Genesis' nonconvex mesh path is SDF-based
}


def geom_exception(geom_type, geom_id, stage_file, reason_msg):
    gs.raise_exception(f"{reason_msg} for {geom_type} {geom_id} in usd file {stage_file}.")


def get_triangle_ids(tri_starts, tri_counts):
    tri_bases = np.repeat(tri_starts, tri_counts)
    tri_offsets = np.arange(tri_counts.sum(), dtype=np.int32)
    tri_stages = np.repeat(np.cumsum(tri_counts, dtype=np.int32) - tri_counts, tri_counts)
    return tri_bases + tri_offsets - tri_stages


def parse_prim_geoms(
    context: UsdContext,
    prim: Usd.Prim,
    link_prim: Usd.Prim,
    links_g_infos: List[List[Dict]],
    link_path_to_idx: Dict[str, int],
    morph: gs.morphs.USD,
    surface: gs.surfaces.Surface,
    match_visual=False,
    match_collision=False,
):
    if not prim.IsActive():
        return

    if str(prim.GetPath()) in link_path_to_idx:
        link_prim = prim

    if not match_visual:
        for pattern in morph.visual_mesh_prim_patterns:
            if re.match(pattern, prim.GetName()):
                match_visual = True
                break
    if not match_collision:
        for pattern in morph.collision_mesh_prim_patterns:
            if re.match(pattern, prim.GetName()):
                match_collision = True
                break

    if link_prim is not None and prim.IsA(UsdGeom.Gprim):
        # parse materials
        geom_surface, geom_uvname, _surface_id, bake_success = context.apply_surface(prim, surface)
        gprim = UsdGeom.Gprim(prim)
        uvs = {geom_uvname: None}

        # parse transform
        geom_Q, geom_S = context.compute_gs_transform(prim, link_prim)
        geom_S *= morph.scale
        geom_ST = np.eye(4, dtype=geom_S.dtype)
        geom_ST[:3, :3] = geom_S
        geom_Q[:3, 3] *= morph.scale
        geom_id = context.get_prim_id(prim)

        # parse geometry
        meshes = []
        axis_T = None  # Set for primitives with axis attribute (Capsule, Cylinder, Plane)
        if prim.IsA(UsdGeom.Mesh):
            mesh_prim = UsdGeom.Mesh(prim)

            # parse vertices
            points = usd_attr_array_to_numpy(mesh_prim.GetPointsAttr(), np.float32)
            if points.size == 0:
                geom_exception("Mesh", geom_id, morph.file, "No vertices")

            # parse faces
            faces = usd_attr_array_to_numpy(mesh_prim.GetFaceVertexIndicesAttr(), np.int32)
            face_vertex_counts = usd_attr_array_to_numpy(mesh_prim.GetFaceVertexCountsAttr(), np.int32)
            points_faces_varying = False

            # parse normals
            normals = usd_attr_array_to_numpy(mesh_prim.GetNormalsAttr(), np.float32, True)
            if normals is not None and normals.shape[0] != points.shape[0]:
                if normals.shape[0] == faces.shape[0]:  # face varying meshes, adjacent faces do not share vertices
                    points_faces_varying = True
                else:
                    gs.logger.warning(
                        f"Normals size mismatch for Mesh {geom_id} in {morph.file}: "
                        f"expected {points.shape[0]} (vertex) or {faces.shape[0]} (faceVarying), "
                        f"got {normals.shape[0]}. Discarding normals for this mesh."
                    )
                    normals = None

            # parse geom subsets
            subset_infos = []
            face_used_mask = np.full(len(face_vertex_counts), False, dtype=np.bool_)
            subsets = UsdGeom.Subset.GetAllGeomSubsets(mesh_prim)
            for subset in subsets:
                subset_prim = subset.GetPrim()
                elem_type = str(subset.GetElementTypeAttr().Get() or "face")
                if str(elem_type) == "face":
                    subset_face_ids_attr = subset.GetIndicesAttr()
                    subset_face_ids = usd_attr_array_to_numpy(subset_face_ids_attr, np.int32)
                    if subset_face_ids.size == 0:
                        continue
                    face_used_mask[subset_face_ids] = True
                    subset_surface, subset_uvname, _, subset_bake_success = context.apply_surface(subset_prim, surface)
                    subset_name = subset_prim.GetPath().pathString
                    subset_infos.append(
                        (subset_face_ids, subset_surface, subset_uvname, subset_name, subset_bake_success)
                    )
                    uvs[subset_uvname] = None
                else:
                    gs.logger.warning(f"Unsupported geom subset element type: {elem_type} for {geom_id}")
            subset_unused = ~face_used_mask
            if subset_unused.any():
                subset_infos.append((subset_unused, geom_surface, geom_uvname, prim.GetPath().pathString, bake_success))

            # parse UVs
            for uvname in uvs.keys():
                uv = usd_primvar_array_to_numpy(UsdGeom.PrimvarsAPI(prim).GetPrimvar(uvname), np.float32, True)
                if uv is not None:
                    uv[:, 1] = 1.0 - uv[:, 1]  # Flip V coordinate
                    if uv.shape[0] != points.shape[0]:
                        if uv.shape[0] == faces.shape[0]:
                            points_faces_varying = True
                        elif uv.shape[0] == 1:
                            uv = None
                        else:
                            gs.logger.warning(
                                f"UV size mismatch for Mesh {geom_id} in {morph.file}: "
                                f"expected {points.shape[0]} (vertex) or {faces.shape[0]} (faceVarying), "
                                f"got {uv.shape[0]}. Discarding UV data for this mesh."
                            )
                            uv = None
                    uvs[uvname] = uv

            # process faces
            if face_vertex_counts.size == 0:
                triangles = np.empty((0, 3), dtype=np.int32)
                face_triangle_starts = np.empty(0, dtype=np.int32)
            else:
                # rearrange points and faces
                if points_faces_varying:
                    if normals is not None and normals.shape[0] == points.shape[0]:
                        normals = normals[faces]
                    for uvname in uvs.keys():
                        uv = uvs[uvname]
                        if uv is not None and uv.shape[0] == points.shape[0]:
                            uvs[uvname] = uv[faces]
                    points = points[faces]
                    faces = np.arange(faces.shape[0], dtype=np.int32)

                # triangulate faces
                # TODO: discard degenerated faces
                if np.max(face_vertex_counts) > 3:
                    triangles, face_triangle_starts = [], []
                    bi, ti = 0, 0
                    for face_vertex_count in face_vertex_counts:
                        face_triangle_starts.append(ti)
                        if face_vertex_count == 3:
                            triangles.append([faces[bi + 0], faces[bi + 1], faces[bi + 2]])
                        elif face_vertex_count > 3:
                            for i in range(1, face_vertex_count - 1):
                                triangles.append([faces[bi + 0], faces[bi + i], faces[bi + i + 1]])
                        bi += face_vertex_count
                        ti += face_vertex_count - 2
                    triangles = np.asarray(triangles, dtype=np.int32)
                    face_triangle_starts = np.asarray(face_triangle_starts, dtype=np.int32)
                else:
                    triangles = faces.reshape(-1, 3)
                    face_triangle_starts = np.arange(len(face_vertex_counts), dtype=np.int32)

            # process mesh
            for subset_face_ids, subset_surface, subset_uvname, subset_name, subset_bake_success in subset_infos:
                tri_starts = face_triangle_starts[subset_face_ids]
                tri_counts = face_vertex_counts[subset_face_ids] - 2
                tri_ids = get_triangle_ids(tri_starts, tri_counts)
                subset_triangles = triangles[tri_ids]
                subset_uv = uvs[subset_uvname]

                processed_mesh = trimesh.Trimesh(
                    vertices=points,
                    faces=subset_triangles,
                    vertex_normals=normals,
                    visual=trimesh.visual.TextureVisuals(uv=subset_uv) if subset_uv is not None else None,
                    process=False,
                )
                processed_mesh.remove_unreferenced_vertices()
                processed_mesh.apply_transform(geom_ST)
                subset_points = processed_mesh.vertices
                subset_triangles = processed_mesh.faces
                subset_normals = processed_mesh.vertex_normals
                if subset_uv is not None:
                    subset_uv = processed_mesh.visual.uv

                # Deduplicate vertices by (position, normal, UV) deterministically using np.unique.
                # This replaces trimesh's process=True which internally calls fix_normals(), causing
                # non-deterministic normal modifications that break cross-format mesh comparison.
                # Round to 8 decimal places to merge near-identical vertices from USD face-varying
                # encoding while preserving truly distinct vertices.
                attrs = [subset_points, subset_normals]
                if subset_uv is not None:
                    attrs.append(subset_uv)
                all_attrs = np.concatenate(attrs, axis=1)
                _, unique_idx, inverse_idx = np.unique(
                    np.round(all_attrs, 8), axis=0, return_index=True, return_inverse=True
                )
                subset_points = subset_points[unique_idx]
                subset_normals = subset_normals[unique_idx]
                if subset_uv is not None:
                    subset_uv = subset_uv[unique_idx]
                subset_triangles = inverse_idx[subset_triangles]

                mesh = gs.Mesh.from_attrs(
                    verts=subset_points,
                    faces=subset_triangles,
                    normals=subset_normals,
                    surface=subset_surface,
                    uvs=subset_uv,
                )
                # Named by prim path, as a link is, so the name holds nothing of the author's filesystem
                mesh.metadata.update(
                    {
                        "mesh_path": context.stage_file,  # unbaked file or cache
                        "name": subset_name,
                        "bake_success": bool(subset_bake_success),
                    }
                )
                meshes.append(mesh)

            geom_data = None
            gs_type = gs.GEOM_TYPE.MESH

        else:  # primitive geometries
            # Reflection from negative xformOp:scale is carried by geom_ST; collision sizes stay positive.
            geom_S_diag = np.abs(np.diag(geom_S))
            if not np.allclose(geom_S_diag, geom_S_diag[0], atol=1e-6):
                gs.logger.warning(
                    f"Non-uniform scale {geom_S_diag} on primitive {prim.GetPath()}. "
                    "Using first axis scale for collision geometry data."
                )
            geom_scale = float(geom_S_diag[0])

            if prim.IsA(UsdGeom.Plane):
                plane_prim = UsdGeom.Plane(prim)
                width = plane_prim.GetWidthAttr().Get()
                length = plane_prim.GetLengthAttr().Get()
                plane_axis_str = plane_prim.GetAxisAttr().Get() or "Z"
                axis_T = AXES_T[plane_axis_str]

                w = float(width) * 0.5
                l = float(length) * 0.5
                tmesh = trimesh.Trimesh(
                    vertices=np.array([[-w, -l, 0.0], [w, -l, 0.0], [w, l, 0.0], [-w, l, 0.0]], dtype=np.float32),
                    faces=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32),
                    face_normals=np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32),
                )
                geom_data = AXES_VECTOR[plane_axis_str]
                gs_type = gs.GEOM_TYPE.PLANE

            elif prim.IsA(UsdGeom.Sphere):
                sphere_prim = UsdGeom.Sphere(prim)
                radius = sphere_prim.GetRadiusAttr().Get()
                tmesh = trimesh.creation.icosphere(radius=radius, subdivisions=2)
                geom_data = np.array([radius * geom_scale])
                gs_type = gs.GEOM_TYPE.SPHERE

            elif prim.IsA(UsdGeom.Capsule):
                capsule_prim = UsdGeom.Capsule(prim)
                radius = capsule_prim.GetRadiusAttr().Get()
                height = capsule_prim.GetHeightAttr().Get()
                axis_T = AXES_T[capsule_prim.GetAxisAttr().Get() or "Z"]
                tmesh = trimesh.creation.capsule(radius=radius, height=height, count=(8, 12))
                geom_data = np.array([radius * geom_scale, height * geom_scale])
                gs_type = gs.GEOM_TYPE.CAPSULE

            elif prim.IsA(UsdGeom.Cube):
                cube_prim = UsdGeom.Cube(prim)
                size = cube_prim.GetSizeAttr().Get()
                extents = np.array([size, size, size])
                tmesh = trimesh.creation.box(extents=extents)
                geom_data = extents * geom_S_diag
                geom_surface.smooth = False
                gs_type = gs.GEOM_TYPE.BOX

            elif prim.IsA(UsdGeom.Cylinder):
                cylinder_prim = UsdGeom.Cylinder(prim)
                radius = cylinder_prim.GetRadiusAttr().Get()
                height = cylinder_prim.GetHeightAttr().Get()
                axis_T = AXES_T[cylinder_prim.GetAxisAttr().Get() or "Z"]
                tmesh = trimesh.creation.cylinder(radius=radius, height=height, count=(8, 12))
                geom_data = np.array([radius * geom_scale, height * geom_scale])
                geom_surface.smooth = False
                gs_type = gs.GEOM_TYPE.CYLINDER

            else:
                gs.raise_exception(f"Unsupported geometry type: {prim.GetTypeName()}")

            # Mesh stays Z-aligned; axis orientation is handled by the quat (matching MuJoCo pattern).
            # axis_T is NOT baked into mesh vertices — it goes into geom_Q instead.
            tmesh.apply_transform(geom_ST)
            metadata = {
                "name": prim.GetPath().pathString,
                "bake_success": bool(bake_success),
            }
            meshes.append(gs.Mesh.from_trimesh(tmesh, surface=geom_surface, metadata=metadata))

        # Compose axis rotation into geom transform for oriented primitives.
        if axis_T is not None:
            geom_Q = geom_Q @ axis_T

        geom_pos = geom_Q[:3, 3]
        geom_quat = gu.R_to_quat(geom_Q[:3, :3])

        is_guide = str(gprim.GetPurposeAttr().Get() or "default") == "guide"
        is_visible = str(gprim.ComputeVisibility()) != "invisible"
        is_visual = (is_visible and not is_guide) and (match_visual or not (match_collision or match_visual))
        is_collision = is_visible and (match_collision or not (match_collision or match_visual))

        g_infos = links_g_infos[link_path_to_idx[str(link_prim.GetPath())]]
        if is_visual:
            for mesh in meshes:
                g_infos.append(
                    dict(
                        vmesh=mesh,
                        pos=geom_pos,
                        quat=geom_quat,
                        contype=0,
                        conaffinity=0,
                        type=gs_type,
                        data=geom_data,
                    )
                )
        if is_collision:
            # A bound UsdPhysicsMaterialAPI overrides the default friction. Prefer dynamic friction,
            # falling back to static; restitution has no rigid-rigid equivalent in the solver and is
            # dropped (with a one-time warning).
            geom_friction = gu.default_friction()
            physics_material = context.get_physics_material(prim)
            if physics_material is not None:
                # PhysicsMaterial fields are None when unauthored, so an explicitly authored 0 (a
                # frictionless collider) is honored. Prefer dynamic friction, fall back to static.
                if physics_material.dynamic_friction is not None:
                    geom_friction = physics_material.dynamic_friction
                elif physics_material.static_friction is not None:
                    geom_friction = physics_material.static_friction
                if physics_material.restitution:
                    context.note_unsupported_restitution()

            # Per-geom collision approximation hint (UsdPhysics.MeshCollisionAPI), honored only when
            # explicitly authored. Meaningful for mesh collision; primitives are already exact.
            approximation = None
            if prim.IsA(UsdGeom.Mesh) and prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                approximation_attr = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr()
                if approximation_attr.HasAuthoredValue():
                    approximation = approximation_attr.Get()

            # 'prim_path' resolves collision-group membership (see usd_collision) and 'density' feeds
            # the density-derived link mass; parse_usd_rigid_entity strips both once consumed.
            collision_g_info = dict(
                pos=geom_pos,
                quat=geom_quat,
                contype=1,
                conaffinity=1,
                friction=geom_friction,
                prim_path=str(prim.GetPath()),
            )
            if physics_material is not None and physics_material.density is not None:
                collision_g_info["density"] = physics_material.density

            if approximation in ("boundingCube", "boundingSphere") and meshes:
                # Fit a single primitive to the collision vertices, replacing the mesh collider.
                verts = np.vstack([mesh.verts for mesh in meshes])
                lo, hi = verts.min(axis=0), verts.max(axis=0)
                center = (lo + hi) * 0.5
                prim_pos = geom_pos + gu.transform_by_quat(center, geom_quat)
                if approximation == "boundingCube":
                    extents = hi - lo
                    bv_tmesh = trimesh.creation.box(extents=extents)
                    bv_type, bv_data = gs.GEOM_TYPE.BOX, extents
                else:
                    radius = np.linalg.norm(verts - center, axis=1).max()
                    bv_tmesh = trimesh.creation.icosphere(radius=radius, subdivisions=2)
                    bv_type, bv_data = gs.GEOM_TYPE.SPHERE, np.array([radius])
                bv_mesh = gs.Mesh.from_trimesh(
                    bv_tmesh, surface=geom_surface, metadata={"name": prim.GetPath().pathString}
                )
                g_infos.append(
                    {
                        **collision_g_info,
                        "mesh": bv_mesh,
                        "pos": prim_pos,
                        "sol_params": gu.default_solver_params(),
                        "type": bv_type,
                        "data": bv_data,
                        # Already a primitive: skip mesh convexify for this geom.
                        "convexify": False,
                    }
                )
            else:
                approximation_overrides = _APPROXIMATION_OVERRIDES.get(approximation, {})
                for mesh in meshes:
                    g_infos.append(
                        {
                            **collision_g_info,
                            "mesh": mesh,
                            "sol_params": gu.default_solver_params(),
                            "type": gs_type,
                            "data": geom_data,
                            **approximation_overrides,
                        }
                    )

    predicate = Usd.TraverseInstanceProxies()
    prim_range = Usd.PrimRange(prim, predicate)
    iterator = iter(prim_range)
    # skip the first prim (current prim)
    next(iterator)
    for child in iterator:
        parse_prim_geoms(
            context, child, link_prim, links_g_infos, link_path_to_idx, morph, surface, match_visual, match_collision
        )
        iterator.PruneChildren()
