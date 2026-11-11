import numpy as np

import igl

import genesis as gs

from . import mesh as mu
from .cache import get_tet_cache


def mesh_to_elements(mesh, tet_cfg=dict()):
    """Tetrahedralize a surface trimesh, with the result cached on disk keyed on vertices, faces and configuration."""
    cache = get_tet_cache(mesh.vertices, mesh.faces, tet_cfg)

    # loading pre-computed cache if available
    elements = cache.load()
    if elements is not None:
        gs.logger.debug("Tetrahedra file (`.tet`) found in cache.")
        return elements

    with gs.logger.timer(f"Tetrahedralization with configuration {tet_cfg} and generating `.tet` file:"):
        elements = mu.tetrahedralize_mesh(mesh, tet_cfg)
        cache.save(elements)
    return elements


def clip_convex_polyhedron(
    polygons: list[np.ndarray], normal: np.ndarray, offset: float, tol: float
) -> list[np.ndarray]:
    """Clip a convex polyhedron, given as the polygons of its faces, by the half-space normal . x <= offset.

    The faces keep their part inside the half-space, and the cut closes with a new face. The points within tol of the
    plane count as on it.
    """
    polygons_clipped, points_cut = [], []
    for polygon in polygons:
        dists = polygon @ normal - offset
        is_inside = dists <= tol
        points_cut += list(polygon[np.abs(dists) <= tol])
        if is_inside.all():
            polygons_clipped.append(polygon)
            continue
        points = []
        for i in range(len(polygon)):
            j = (i + 1) % len(polygon)
            if is_inside[i]:
                points.append(polygon[i])
            if is_inside[i] != is_inside[j] and np.abs(dists[i]) > tol and np.abs(dists[j]) > tol:
                point = polygon[i] + dists[i] / (dists[i] - dists[j]) * (polygon[j] - polygon[i])
                points.append(point)
                points_cut.append(point)
        if len(points) >= 3:
            polygons_clipped.append(np.stack(points))
    # The cut, its points in turn around their center in the plane
    if len(points_cut) >= 3:
        points_cut = np.stack(points_cut)
        # The points the polygons share, taken once each without moving them
        _, points_idx = np.unique(np.round(points_cut / tol), axis=0, return_index=True)
        points_cut = points_cut[np.sort(points_idx)]
        center = points_cut.mean(axis=0)
        axis_u = np.cross(normal, np.eye(3)[np.argmin(np.abs(normal))])
        axis_v = np.cross(normal, axis_u)
        angles = np.arctan2((points_cut - center) @ axis_v, (points_cut - center) @ axis_u)
        if len(points_cut) >= 3:
            polygons_clipped.append(points_cut[np.argsort(angles)])
    return polygons_clipped


def convex_mesh_to_elements(verts: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Tetrahedralize a convex surface mesh so that the distance to its surface is linear over every tet.

    The interior splits into the regions closest to each plane of the faces, which the medial axis bounds, the faces of
    one plane merged. Each region splits into layers parallel to its plane, about as thick as the edges of its faces, so
    that its tets stay compact. The deepest layer fans into tets from the center of the vertices of the region deepest
    below its plane, on the medial axis, which keeps the symmetries of the region, the others from their center. The
    triangles of the layer at the surface split into 4 at the midpoints of their edges, so that the slices of a face
    against a plane spread over it rather than along a diagonal. A linear field over the tets then holds the distance to
    the surface exactly. The tets of two neighboring regions may cut their shared boundary differently. The vertices come
    first, then the tets, positively oriented, as indices into them.
    """
    tris = verts[faces]
    normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
    offsets = np.einsum("ij,ij->i", normals, tris[:, 0])
    # A few roundings of the coordinates, relative to the size of the mesh
    tol = 8.0 * np.finfo(np.float64).eps * np.linalg.norm(np.ptp(verts, axis=0))

    # The faces of one plane, edge-adjacent ones whose vertices lie on the plane of each other, take the plane of their
    # lowest face, the planes being those of the faces of the mesh and none other
    edges = np.sort(faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
    edges_face = np.repeat(np.arange(len(faces)), 3)
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    is_shared = (edges[order][1:] == edges[order][:-1]).all(axis=-1)
    pairs = np.stack([edges_face[order][:-1][is_shared], edges_face[order][1:][is_shared]], axis=-1)
    pairs_dist = np.einsum("ijk,ik->ij", tris[pairs[:, 1]], normals[pairs[:, 0]]) - offsets[pairs[:, 0], None]
    pairs = pairs[(np.abs(pairs_dist) <= tol).all(axis=-1)]
    faces_plane_idx = np.arange(len(faces))
    while True:
        faces_plane_idx_next = faces_plane_idx.copy()
        np.minimum.at(faces_plane_idx_next, pairs[:, 0], faces_plane_idx[pairs[:, 1]])
        np.minimum.at(faces_plane_idx_next, pairs[:, 1], faces_plane_idx[pairs[:, 0]])
        if (faces_plane_idx_next == faces_plane_idx).all():
            break
        faces_plane_idx = faces_plane_idx_next
    planes_face_idx = np.unique(faces_plane_idx)
    planes_normal, planes_offset = normals[planes_face_idx], offsets[planes_face_idx]

    # The box bounding the mesh, as its faces, from which each region is clipped
    lower, upper = verts.min(axis=0), verts.max(axis=0)
    box_corners = np.stack(np.meshgrid(*np.stack([lower, upper], axis=-1), indexing="ij"), axis=-1).reshape(-1, 3)
    box = [
        box_corners[list(face)]
        for face in ((0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3))
    ]
    # The mean length of the edges of the faces of each plane, the thickness of the layers of its region
    faces_edge_length = np.linalg.norm(tris - np.roll(tris, -1, axis=1), axis=-1).mean(axis=-1)
    planes_edge_length = (
        np.bincount(faces_plane_idx, weights=faces_edge_length)[planes_face_idx]
        / np.bincount(faces_plane_idx)[planes_face_idx]
    )

    regions_tets = []
    for i_p in range(len(planes_face_idx)):
        # The box clipped by the half-spaces inside the mesh and closer to the plane than to each other one, the most
        # violated first
        halfspaces_normal = np.concatenate([planes_normal, np.delete(planes_normal, i_p, axis=0) - planes_normal[i_p]])
        halfspaces_offset = np.concatenate([planes_offset, np.delete(planes_offset, i_p) - planes_offset[i_p]])
        region = box
        while True:
            violations = (np.concatenate(region) @ halfspaces_normal.T - halfspaces_offset).max(axis=0)
            i_q = np.argmax(violations)
            if violations[i_q] <= tol:
                break
            region = clip_convex_polyhedron(region, halfspaces_normal[i_q], halfspaces_offset[i_q], tol)

        # The layers of the region parallel to its plane, as thick as the edges of its faces, so that its tets stay as
        # compact. The deepest one fans from the center of the deepest vertices, on the medial axis, the others from
        # their center, over the faces away from it, each face triangulated from its first vertex. The triangles of the
        # layer at the surface split into 4, so that the slices of its face against a plane spread over it.
        depths = planes_offset[i_p] - np.concatenate(region) @ planes_normal[i_p]
        n_layers = max(1, int(np.ceil(depths.max() / planes_edge_length[i_p] - 0.5)))
        layer_thickness = depths.max() / n_layers
        for i_layer in range(n_layers):
            layer = region
            if i_layer > 0:
                layer = clip_convex_polyhedron(
                    layer, planes_normal[i_p], planes_offset[i_p] - i_layer * layer_thickness, tol
                )
            if i_layer < n_layers - 1:
                layer = clip_convex_polyhedron(
                    layer, -planes_normal[i_p], (i_layer + 1) * layer_thickness - planes_offset[i_p], tol
                )
            layer_verts = np.concatenate(layer)
            if i_layer == n_layers - 1:
                heights = layer_verts @ planes_normal[i_p]
                apex = layer_verts[heights <= heights.min() + tol].mean(axis=0)
            else:
                apex = np.unique(np.round(layer_verts / tol), axis=0).mean(axis=0) * tol
            fan_tris = []
            for polygon in layer:
                polygon_normal = np.cross(polygon[1:-1] - polygon[0], polygon[2:] - polygon[0]).sum(axis=0)
                if np.abs((apex - polygon[0]) @ polygon_normal) > tol * np.linalg.norm(polygon_normal):
                    fan_tris += [np.stack([polygon[0], polygon[i], polygon[i + 1]]) for i in range(1, len(polygon) - 1)]
            fan_tris = np.stack(fan_tris)
            if i_layer == 0:
                mids = 0.5 * (fan_tris + np.roll(fan_tris, -1, axis=1))
                fan_tris = np.concatenate(
                    [
                        np.stack([fan_tris[:, 0], mids[:, 0], mids[:, 2]], axis=1),
                        np.stack([mids[:, 0], fan_tris[:, 1], mids[:, 1]], axis=1),
                        np.stack([mids[:, 2], mids[:, 1], fan_tris[:, 2]], axis=1),
                        mids,
                    ]
                )
            regions_tets.append(np.concatenate([np.broadcast_to(apex, (len(fan_tris), 1, 3)), fan_tris], axis=1))

    # Every tet positively oriented, the flat ones of degenerate faces left out, the regions sharing the vertices of
    # their common boundary
    tets = np.concatenate(regions_tets)
    edges = tets[:, 1:] - tets[:, :1]
    volumes = np.einsum("ij,ij->i", np.cross(edges[:, 0], edges[:, 1]), edges[:, 2])
    # A tet is flat once its volume relative to its edges is below the square root of the precision
    is_solid = np.abs(volumes) > np.sqrt(np.finfo(np.float64).eps) * np.linalg.norm(edges, axis=-1).prod(axis=-1)
    tets, volumes = tets[is_solid], volumes[is_solid]
    tets[volumes < 0.0, 1:3] = tets[volumes < 0.0, 2:0:-1]
    verts, verts_idx = np.unique(np.round(tets.reshape(-1, 3) / tol), axis=0, return_inverse=True)
    elems = verts_idx.reshape(-1, 4)
    # A tet whose vertices merge within the tolerance has no volume left
    elems_sorted = np.sort(elems, axis=-1)
    return verts * tol, elems[(elems_sorted[:, 1:] != elems_sorted[:, :-1]).all(axis=-1)]


def split_all_surface_tets(verts, elems):
    """
    Splits tetrahedras that have 4 vertices on the surface into 4 smaller tetrahedras.

    This is useful for the hydroelastic contact model.
    """
    F, *_ = igl.boundary_facets(elems)
    on_surface = np.zeros(verts.shape[0], dtype=bool)
    on_surface[F.reshape(-1)] = True
    all_on_surface = np.all(on_surface[elems], axis=1)
    if not all_on_surface.any():
        return verts, elems
    bad_elems = elems[all_on_surface]
    new_verts = np.mean(verts[bad_elems], axis=1, dtype=np.float32)
    new_elems = []
    for idx, (v0, v1, v2, v3) in enumerate(bad_elems, len(verts)):
        new_elems.append([v0, v1, v2, idx])
        new_elems.append([v0, v1, idx, v3])
        new_elems.append([v0, idx, v2, v3])
        new_elems.append([idx, v1, v2, v3])
    new_elems = np.array(new_elems, dtype=np.int32)
    verts = np.concatenate([verts, new_verts], axis=0)
    # remove the bad elements from the original elements
    elems = np.concatenate([elems[~all_on_surface], new_elems], axis=0)
    return verts, elems
