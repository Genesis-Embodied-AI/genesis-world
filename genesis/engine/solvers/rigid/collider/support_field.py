import math
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial import ConvexHull

import quadrants as qd

import genesis as gs
import genesis.utils.array_class as array_class
import genesis.utils.geom as gu


if TYPE_CHECKING:
    from genesis.engine.solvers.rigid.rigid_solver import RigidSolver


# Rotation from the mesh frame to the frame of each spherical chart. The second chart carries its poles a quarter turn
# away from those of the first, onto +/- y (see _func_direction_cell).
CHARTS_ROT = np.stack((np.eye(3), gu.euler_to_R(np.array((90.0, 0.0, 0.0)))))


class SupportField:
    """
    Support table of the geoms: for every cell of a grid of directions, the vertices supporting a direction of the cell.

    A lookup scans the list of the cell its direction falls in, which makes it exact. The normal cone of a vertex (the
    directions it supports) meets a cell only by holding one of its corners, by holding the normal of one of its faces
    inside the cell, or by crossing the cell with one of its edges, which are the arcs between the normals of two
    adjacent hull faces. A cell therefore lists the supports of its four corners, the vertices of the hull faces whose
    normal it holds, and the two vertices of every hull edge whose arc crosses it.

    The grid is made of two spherical charts, each one answering for the band of directions 45 to 135 degrees off its
    own poles (see _func_direction_cell), so only the cells of that band are stored.

    The first two candidates of every cell are stored in a record at the index of the cell, and the others in a list
    apart. Most cells have at most two candidates, whose reads then issue at once, ahead of the read of the start of
    their list. A lookup runs on the critical path of collision detection, where that wait costs more than the reads.
    """

    def __init__(self, rigid_solver: "RigidSolver") -> None:
        self.solver = rigid_solver
        # The table is exact at any resolution, which only trades its memory against the length of the lists of its
        # cells. A multiple of four makes the band of rows of each chart span 45 to 135 degrees off its poles exactly.
        self._support_res = 120
        self._support_field_info = array_class.get_support_field_info(0, 0, 0, self._support_res)
        self._is_active = False

    def activate(self) -> None:
        if self.is_active:
            return

        band_start, band_rows = self._support_res // 4, self._support_res // 2
        n_geom_cells = 2 * self._support_res * band_rows

        # Unit direction in the mesh frame of every node bounding the cells of the charts, as [chart, azimuth, polar]
        theta = np.arange(self._support_res) / self._support_res * 2 * math.pi - math.pi
        phi = np.arange(band_start, band_start + band_rows + 1) / self._support_res * math.pi
        nodes_chart = np.stack(
            np.broadcast_arrays(
                np.sin(phi) * np.cos(theta[:, None]), np.sin(phi) * np.sin(theta[:, None]), np.cos(phi)
            ),
            axis=-1,
        )
        nodes = nodes_chart @ CHARTS_ROT[:, None]

        # Four corner nodes of each cell
        cells_chart, cells_i, cells_j, corners_di, corners_dj = np.meshgrid(
            np.arange(2), np.arange(self._support_res), np.arange(band_rows), (0, 1), (0, 1), indexing="ij"
        )
        cells_corner = np.ravel_multi_index(
            (cells_chart, (cells_i + corners_di) % self._support_res, cells_j + corners_dj), nodes.shape[:3]
        ).reshape((n_geom_cells, 4))

        # Planes of the meridians and polar angles of the parallels bounding the cells of a chart, in its own frame
        meridians_theta = theta[: self._support_res // 2]
        meridians_normal = np.stack(
            (-np.sin(meridians_theta), np.cos(meridians_theta), np.zeros_like(meridians_theta)), axis=-1
        )
        parallels_z = np.cos(phi)
        # Corners of the closure of a cell around a direction, widened by a margin in cell units
        cells_corner_margin = 1e-6 * np.array(((-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0)))

        geoms_cell_start = []
        cells_count = []
        cells_vid = []
        cells_v = []
        extras_vid = []
        extras_v = []
        n_support_cells = 0
        for geom in self.solver.geoms:
            geoms_cell_start.append(n_support_cells)

            # The support of a terrain is read off the prism built from its height field, so its table would never
            # be read. Its vertex count (one per height field sample) would make it the costliest table to build.
            if geom.type == gs.GEOM_TYPE.TERRAIN:
                continue

            # Normal cones of the hull vertices, as the faces and edges of their union: the normals of the hull faces,
            # each supported by the vertices of its face, and the arcs between the normals of adjacent hull faces, along
            # which both vertices of their shared edge tie. A flat geom joins its two opposite face normals through the
            # outward normal of each outline edge by two quarter arcs. A segment or a point ties both of its ends along
            # the great circle orthogonal to it, cut in four quarter arcs.
            verts = geom.init_verts
            verts_centered = verts - verts.mean(axis=0)
            verts_rank = np.linalg.matrix_rank(verts_centered)
            # Principal axes of the vertices, from the eigenvectors of their 3x3 second moment, of fixed size
            _, _, frame = np.linalg.svd(verts_centered.T @ verts_centered)
            if verts_rank == 3:
                hull = ConvexHull(verts_centered)
                hull_vid = hull.vertices
                faces_idx, i_vert = np.nonzero(hull.neighbors > np.arange(len(hull.simplices))[:, None])
                faces_normal = hull.equations[:, :3]
                faces_direction = np.repeat(faces_normal, repeats=3, axis=0)
                faces_vid = hull.simplices.reshape(-1)
                arcs_start = faces_normal[faces_idx]
                arcs_end = faces_normal[hull.neighbors[faces_idx, i_vert]]
                # The neighbour of a face across its i-th vertex shares the edge made of its two other vertices
                arcs_vid = np.stack(
                    (hull.simplices[faces_idx, (i_vert + 1) % 3], hull.simplices[faces_idx, (i_vert + 2) % 3]), axis=-1
                )
            elif verts_rank == 2:
                outline = ConvexHull(verts_centered @ frame[:2].T)
                hull_vid = outline.vertices
                edges_normal = outline.equations[:, :2] @ frame[:2]
                faces_normal = np.stack((frame[2], -frame[2]))
                n_edges = len(edges_normal)
                faces_direction = np.repeat(faces_normal, repeats=len(hull_vid), axis=0)
                faces_vid = np.tile(hull_vid, reps=2)
                arcs_start = np.concatenate((np.tile(faces_normal[0], reps=(n_edges, 1)), edges_normal))
                arcs_end = np.concatenate((edges_normal, np.tile(faces_normal[1], reps=(n_edges, 1))))
                arcs_vid = np.tile(outline.simplices, reps=(2, 1))
            else:
                verts_axial = verts_centered @ frame[0]
                hull_vid = np.array((np.argmin(verts_axial), np.argmax(verts_axial)))
                faces_direction = np.zeros((0, 3))
                faces_vid = np.zeros((0,), dtype=gs.np_int)
                arcs_start = np.stack((frame[1], frame[2], -frame[1], -frame[2]))
                arcs_end = np.roll(arcs_start, shift=-1, axis=0)
                arcs_vid = np.tile(hull_vid, reps=(4, 1))

            # Support of every node, searched among the hull vertices alone, by chunks of nodes whose dot products take
            # a bounded memory whatever the vertex count
            nodes_chunks = np.array_split(nodes.reshape((-1, 3)), max(nodes[..., 0].size * len(hull_vid) // 2**24, 1))
            nodes_vid = np.concatenate(
                [hull_vid[np.argmax(nodes_chunk @ verts[hull_vid].T, axis=-1)] for nodes_chunk in nodes_chunks]
            )

            # Every arc is cut where it crosses the meridians and parallels of a chart, so that each piece lies in a
            # single cell, the cell of its midpoint. A crossing outside of the arc is clamped to one of its ends. The
            # cuts lie on the boundaries of the cells, which an arc tangent to a parallel touches without crossing, so
            # they are listed along with the midpoints.
            arcs_cos = np.clip(np.sum(arcs_start * arcs_end, axis=-1), -1.0, 1.0)
            arcs_angle = np.arccos(arcs_cos)
            arcs_orthogonal = arcs_end - arcs_cos[:, None] * arcs_start
            arcs_orthogonal_norm = np.linalg.norm(arcs_orthogonal, axis=-1, keepdims=True)
            arcs_orthogonal = np.divide(
                arcs_orthogonal,
                arcs_orthogonal_norm,
                out=np.zeros_like(arcs_orthogonal),
                where=arcs_orthogonal_norm > 0,
            )

            # The corners of a cell mostly share their support, which is listed once
            corners_vid = nodes_vid[cells_corner]
            is_repeat = np.tril(corners_vid[:, :, None] == corners_vid[:, None, :], k=-1).any(axis=-1)
            corners_cell, i_corner = np.nonzero(~is_repeat)
            pairs = [np.stack((corners_cell, corners_vid[corners_cell, i_corner]), axis=-1)]
            for i_chart in range(2):
                arcs_start_chart = arcs_start @ CHARTS_ROT[i_chart].T
                arcs_orthogonal_chart = arcs_orthogonal @ CHARTS_ROT[i_chart].T
                arcs_meridian = np.mod(
                    np.arctan2(-arcs_start_chart @ meridians_normal.T, arcs_orthogonal_chart @ meridians_normal.T),
                    math.pi,
                )
                arcs_amplitude = np.hypot(arcs_start_chart[:, 2], arcs_orthogonal_chart[:, 2])
                arcs_phase = np.arctan2(arcs_orthogonal_chart[:, 2], arcs_start_chart[:, 2])
                arcs_ratio = parallels_z / np.maximum(arcs_amplitude, gs.EPS)[:, None]
                # A parallel the great circle of an arc misses, or touches up to rounding, cuts it at its extreme point
                arcs_half = np.arccos(np.clip(arcs_ratio, -1.0, 1.0))
                arcs_cuts = np.concatenate(
                    (
                        np.zeros_like(arcs_angle)[:, None],
                        arcs_meridian,
                        np.mod(arcs_phase[:, None] + arcs_half, 2 * math.pi),
                        np.mod(arcs_phase[:, None] - arcs_half, 2 * math.pi),
                        arcs_angle[:, None],
                    ),
                    axis=-1,
                )
                arcs_cuts = np.sort(np.minimum(arcs_cuts, arcs_angle[:, None]), axis=-1)
                arcs_idx, i_cut = np.nonzero(arcs_cuts[:, 1:] > arcs_cuts[:, :-1])
                # The midpoint and the start of every piece, the starts covering every distinct cut but the end of the
                # arc, a face normal listed with its face
                points_angle = np.concatenate(
                    (0.5 * (arcs_cuts[arcs_idx, i_cut] + arcs_cuts[arcs_idx, i_cut + 1]), arcs_cuts[arcs_idx, i_cut])
                )
                points_arc_idx = np.concatenate((arcs_idx, arcs_idx))
                points_direction = (
                    np.cos(points_angle)[:, None] * arcs_start[points_arc_idx]
                    + np.sin(points_angle)[:, None] * arcs_orthogonal[points_arc_idx]
                )

                # Every face normal lists the vertices of its face, and every point of arc both ends of its hull edge,
                # in the cells whose closure holds it. A direction on a cell boundary belongs to every cell around it,
                # since the vertices it supports may support directions of any of them. The closure is widened by a
                # margin covering the rounding of the cell coordinates, which lists a vertex in one cell more at most.
                for directions, directions_vid in (
                    (faces_direction, faces_vid[:, None]),
                    (points_direction, arcs_vid[points_arc_idx]),
                ):
                    directions_chart = directions @ CHARTS_ROT[i_chart].T
                    theta_chart = np.arctan2(directions_chart[:, 1], directions_chart[:, 0])
                    phi_chart = np.arccos(np.clip(directions_chart[:, 2], -1.0, 1.0))
                    cells_coord = np.stack(
                        (
                            (theta_chart + math.pi) / (2 * math.pi) * self._support_res,
                            phi_chart / math.pi * self._support_res,
                        ),
                        axis=-1,
                    )
                    cells_ij = np.floor(cells_coord[:, None] + cells_corner_margin)
                    cells_i = cells_ij[..., 0].astype(gs.np_int) % self._support_res
                    cells_j = cells_ij[..., 1].astype(gs.np_int) - band_start
                    is_in_band = (cells_j >= 0) & (cells_j < band_rows)
                    directions_idx, _ = np.nonzero(is_in_band)
                    cells_idx = ((i_chart * self._support_res + cells_i) * band_rows + cells_j)[is_in_band]
                    cells_idx = np.repeat(cells_idx, repeats=directions_vid.shape[-1])
                    pairs.append(np.stack((cells_idx, directions_vid[directions_idx].reshape(-1)), axis=-1))

            # Each pair is encoded as one integer key, cells major, so that a flat sort removes the repeated ones
            pairs = np.concatenate(pairs)
            cells_idx, verts_idx = np.divmod(np.unique(pairs[:, 0] * len(verts) + pairs[:, 1]), len(verts))
            # Every cell lists the supports of its corners, so the candidates of each cell come in a row, in cell order.
            # Records are laid out as described in SupportFieldInfo.
            candidates_idx = np.arange(len(cells_idx))
            is_first = np.concatenate(((True,), cells_idx[1:] != cells_idx[:-1]))
            candidates_rank = candidates_idx - np.maximum.accumulate(np.where(is_first, candidates_idx, 0))
            slots_vid = np.repeat(verts_idx[is_first, None], repeats=2, axis=-1)
            is_slot = candidates_rank < 2
            slots_vid[cells_idx[is_slot], candidates_rank[is_slot]] = verts_idx[is_slot]
            cells_vid.append(slots_vid.reshape(-1))
            cells_v.append(verts[slots_vid.reshape(-1)])
            extras_vid.append(verts_idx[~is_slot])
            extras_v.append(verts[verts_idx[~is_slot]])
            cells_count.append(np.bincount(cells_idx[~is_slot], minlength=n_geom_cells))
            n_support_cells += n_geom_cells

        support_list_start = 2 * n_support_cells + np.cumsum(np.concatenate(((0,), *cells_count)), dtype=gs.np_int)
        support_vid = np.concatenate((np.zeros((0,), dtype=gs.np_int), *cells_vid, *extras_vid), dtype=gs.np_int)
        support_v = np.concatenate((np.zeros((0, 3)), *cells_v, *extras_v), dtype=gs.np_float)

        self._support_field_info = array_class.get_support_field_info(
            self.solver.n_geoms, n_support_cells, len(support_vid), self._support_res
        )
        _kernel_init_support(
            np.array(geoms_cell_start, dtype=gs.np_int),
            support_list_start,
            support_v,
            support_vid,
            self._support_field_info,
            self.solver.rigid_config,
        )

        self._is_active = True

    @property
    def is_active(self):
        return self._is_active


@qd.kernel
def _kernel_init_support(
    support_cell_start: qd.types.ndarray(),
    support_list_start: qd.types.ndarray(),
    support_v: qd.types.ndarray(),
    support_vid: qd.types.ndarray(),
    support_field_info: array_class.SupportFieldInfo,
    rigid_config: qd.template(),
):
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_g in range(support_cell_start.shape[0]):
        support_field_info.support_cell_start[i_g] = support_cell_start[i_g]

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_cell in range(support_list_start.shape[0]):
        support_field_info.support_list_start[i_cell] = support_list_start[i_cell]

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_s in range(support_v.shape[0]):
        support_field_info.support_vid[i_s] = support_vid[i_s]
        for k in qd.static(range(3)):
            support_field_info.support_v[i_s][k] = support_v[i_s, k]


@qd.func
def _func_support_world(
    i_g: int,
    d: qd.types.vector(3),
    pos: qd.types.vector(3),
    quat: qd.types.vector(4),
    dyn_info: array_class.DynInfo,
    collider_info: array_class.ColliderInfo,
    exhaustive: qd.template() = False,
):
    """Support position for a world direction.

    The vertex comes from the precomputed support table, or from an exhaustive scan of the mesh vertices when
    'exhaustive' is set. Both are exact, but the scan returns the first of tied vertices at a cost linear in the
    vertex count, while the table returns any of them.
    """
    v = qd.Vector.zero(gs.qd_float, 3)
    v_ = qd.Vector.zero(gs.qd_float, 3)
    vid = 0
    if qd.static(exhaustive):
        d_mesh = gu.qd_transform_by_quat(d, gu.qd_inv_quat(quat))
        v_, vid = _func_support_mesh_exhaustive(i_g, d_mesh, dyn_info)
        v = gu.qd_transform_by_trans_quat(v_, pos, quat)
    else:
        d_mesh = gu.qd_transform_by_quat_fast(d, gu.qd_inv_quat(quat))
        v_, vid = _func_support_mesh(i_g, d_mesh, collider_info)
        v = gu.qd_transform_by_trans_quat_fast(v_, pos, quat)
    return v, v_, vid


@qd.func
def _func_direction_cell(i_g: int, d_mesh: qd.types.vector(3), collider_info: array_class.ColliderInfo):
    """
    Cell of the support table of a geom that a mesh-frame direction falls in.

    The chart whose polar axis the direction is least aligned with answers, which leaves it at least 45 degrees off
    that chart's poles since the smaller of the two alignments never exceeds one over root two. The azimuth is then
    always well conditioned, so a direction along one of the geom's own axes - the normal of a face resting flat,
    whose support vertices tie - is resolved by a chart that has no degeneracy there, and a geom and any rotated copy
    of it pick the same vertex.
    """
    # Band of cells of each chart: see SupportField
    support_res = collider_info.support_field.support_res[None]
    band_start, band_rows = support_res // 4, support_res // 2

    i_chart = gs.qd_int(0)
    d_chart = d_mesh
    if qd.abs(d_mesh[2]) > qd.abs(d_mesh[1]):
        # The first chart's poles sit on +/- z, so a direction closer to them is handed to the second chart, whose
        # own coordinates see it a quarter turn away (see CHARTS_ROT).
        i_chart = 1
        d_chart = qd.Vector([d_mesh[0], -d_mesh[2], d_mesh[1]], dt=gs.qd_float)

    theta = qd.atan2(d_chart[1], d_chart[0])  # [-pi, pi]
    phi = qd.acos(d_chart[2])  # [0, pi]
    i = gs.qd_int(qd.math.floor((theta + math.pi) / math.pi / 2 * support_res)) % support_res
    j = gs.qd_int(qd.math.clamp(qd.math.floor(phi / math.pi * support_res) - band_start, 0, band_rows - 1))
    return collider_info.support_field.support_cell_start[i_g] + (i_chart * support_res + i) * band_rows + j


@qd.func
def _func_support_mesh(i_g: int, d_mesh: qd.types.vector(3), collider_info: array_class.ColliderInfo):
    """Support vertex of a mesh-frame direction, the best of the candidates listed by the cell it falls in."""
    i_cell = _func_direction_cell(i_g, d_mesh, collider_info)
    # Record of the cell (see SupportField)
    v = collider_info.support_field.support_v[2 * i_cell]
    vid = collider_info.support_field.support_vid[2 * i_cell]
    pos = collider_info.support_field.support_v[2 * i_cell + 1]
    dot_max = v.dot(d_mesh)
    dot = pos.dot(d_mesh)
    if dot > dot_max:
        v = pos
        dot_max = dot
        vid = collider_info.support_field.support_vid[2 * i_cell + 1]
    # The other candidates are read four at a time, so that the reads of a chunk issue together. The index is clamped to
    # the end of the list, which repeats its last candidate, unable to win twice.
    i_s_start = collider_info.support_field.support_list_start[i_cell]
    i_s_end = collider_info.support_field.support_list_start[i_cell + 1]
    for i_chunk_ in range((i_s_end - i_s_start + 3) // 4):
        for i_k in qd.static(range(4)):
            i_s = qd.min(i_s_start + 4 * i_chunk_ + i_k, i_s_end - 1)
            pos = collider_info.support_field.support_v[i_s]
            dot = pos.dot(d_mesh)
            if dot > dot_max:
                v = pos
                dot_max = dot
                vid = collider_info.support_field.support_vid[i_s]

    return v, vid


@qd.func
def _func_support_mesh_exhaustive(i_g: int, d_mesh: qd.types.vector(3), dyn_info: array_class.DynInfo):
    """Support vertex by exhaustive scan in the mesh frame, the first maximum winning, like the scan the reference
    engine's fallback collision pipeline runs below its hill-climb threshold."""
    dot_max = gs.qd_float(-1e20)
    i_max = dyn_info.geoms.vert_start[i_g]
    for i_v in range(dyn_info.geoms.vert_start[i_g], dyn_info.geoms.vert_end[i_g]):
        vdot = d_mesh.dot(dyn_info.verts.init_pos[i_v])
        if vdot > dot_max:
            dot_max = vdot
            i_max = i_v
    return dyn_info.verts.init_pos[i_max], i_max


@qd.func
def _func_support_sphere(
    i_g: int,
    d: qd.types.vector(3),
    pos: qd.types.vector(3),
    quat: qd.types.vector(4),
    dyn_info: array_class.DynInfo,
    shrink: bool,
):
    sphere_center = pos
    sphere_radius = dyn_info.geoms.data[i_g][0]

    # Shrink the sphere to a point
    v = sphere_center
    v_ = qd.Vector.zero(gs.qd_float, 3)
    vid = -1
    if not shrink:
        v += d * sphere_radius

        # Local position of the support point
        local_d = gu.qd_inv_transform_by_quat(d, quat)
        v_ = local_d * sphere_radius

    return v, v_, vid


@qd.func
def _func_support_ellipsoid(
    i_g: int, d: qd.types.vector(3), pos: qd.types.vector(3), quat: qd.types.vector(4), dyn_info: array_class.DynInfo
):
    a = dyn_info.geoms.data[i_g][0]
    b = dyn_info.geoms.data[i_g][1]
    c = dyn_info.geoms.data[i_g][2]

    # Transform direction to ellipsoid local frame
    d_local = gu.qd_inv_transform_by_quat(d, quat)

    # Support point in local frame: diag(a^2, b^2, c^2) * d_local / |diag(a, b, c) * d_local|
    sx = a * a * d_local[0]
    sy = b * b * d_local[1]
    sz = c * c * d_local[2]
    norm = qd.sqrt(a * a * d_local[0] ** 2 + b * b * d_local[1] ** 2 + c * c * d_local[2] ** 2)
    s_local = qd.Vector([sx / norm, sy / norm, sz / norm], dt=gs.qd_float)

    return pos + gu.qd_transform_by_quat(s_local, quat)


@qd.func
def _func_support_capsule(
    i_g: int,
    d: qd.types.vector(3),
    pos: qd.types.vector(3),
    quat: qd.types.vector(4),
    dyn_info: array_class.DynInfo,
    shrink: bool,
):
    """
    Support function for capsule geometry.

    Thread-safety note: Fully migrated to use explicit pos/quat parameters.
    The i_g parameter is only used for read-only metadata access (radius, halflength)
    from geoms_info, which is thread-safe. Does not access geoms_state.
    """
    res = gs.qd_vec3(0, 0, 0)
    capsule_center = pos
    capsule_radius = dyn_info.geoms.data[i_g][0]
    capsule_halflength = 0.5 * dyn_info.geoms.data[i_g][1]

    if shrink:
        local_dir = gu.qd_transform_by_quat(d, gu.qd_inv_quat(quat))
        res[2] = capsule_halflength if local_dir[2] >= 0.0 else -capsule_halflength
        res = gu.qd_transform_by_trans_quat(res, capsule_center, quat)
    else:
        capsule_axis = gu.qd_transform_by_quat(qd.Vector([0.0, 0.0, 1.0], dt=gs.qd_float), quat)
        capsule_endpoint_side = -1.0 if d.dot(capsule_axis) < 0.0 else 1.0
        capsule_endpoint = capsule_center + capsule_halflength * capsule_endpoint_side * capsule_axis
        res = capsule_endpoint + d * capsule_radius
    return res


@qd.func
def _func_support_cylinder(
    i_g: int,
    d: qd.types.vector(3),
    pos: qd.types.vector(3),
    quat: qd.types.vector(4),
    dyn_info: array_class.DynInfo,
    shrink: bool,
):
    """
    Support function for cylinder geometry.

    Like the capsule, but with flat caps: the support point is on the rim of the cap selected by the sign of d along
    the axis, displaced radially by the radius along d projected onto the cap plane (a sphere/hemisphere cap would
    instead displace along d itself). When d is axial the radial part vanishes and the support is the cap centre.
    """
    radius = dyn_info.geoms.data[i_g][0]
    halflength = 0.5 * dyn_info.geoms.data[i_g][1]
    axis = gu.qd_transform_by_quat(qd.Vector([0.0, 0.0, 1.0], dt=gs.qd_float), quat)
    endpoint_side = -1.0 if d.dot(axis) < 0.0 else 1.0
    res = pos + halflength * endpoint_side * axis
    if not shrink:
        d_radial = d - d.dot(axis) * axis
        d_radial_norm = d_radial.norm()
        if d_radial_norm > 1e-9:
            res = res + (radius / d_radial_norm) * d_radial
    return res


@qd.func
def _func_support_prism(i_b: int, d: qd.types.vector(3), collider_state: array_class.ColliderState):
    istart = 3
    if d[2] < 0:
        istart = 0

    ibest = istart
    best = collider_state.prism[istart, i_b].dot(d)
    for i in range(istart + 1, istart + 3):
        dot = collider_state.prism[i, i_b].dot(d)
        if dot > best:
            ibest = i
            best = dot

    return collider_state.prism[ibest, i_b], ibest


@qd.func
def _func_support_box(
    i_g: int, d: qd.types.vector(3), pos: qd.types.vector(3), quat: qd.types.vector(4), dyn_info: array_class.DynInfo
):
    d_box = gu.qd_inv_transform_by_quat(d, quat)

    v_ = qd.Vector(
        [
            (-1.0 if d_box[0] < 0.0 else 1.0) * dyn_info.geoms.data[i_g][0] * 0.5,
            (-1.0 if d_box[1] < 0.0 else 1.0) * dyn_info.geoms.data[i_g][1] * 0.5,
            (-1.0 if d_box[2] < 0.0 else 1.0) * dyn_info.geoms.data[i_g][2] * 0.5,
        ],
        dt=gs.qd_float,
    )
    vid = (v_[0] > 0.0) * 1 + (v_[1] > 0.0) * 2 + (v_[2] > 0.0) * 4
    vid += dyn_info.geoms.vert_start[i_g]
    v = gu.qd_transform_by_trans_quat_fast(v_, pos, quat)
    return v, v_, vid


@qd.func
def _func_count_supports_world(
    i_g: int, d: qd.types.vector(3), quat: qd.types.vector(4), collider_info: array_class.ColliderInfo
):
    """
    Count the number of valid support points for the given world direction.
    Only needs quat since counting doesn't depend on position.
    """
    d_mesh = gu.qd_transform_by_quat_fast(d, gu.qd_inv_quat(quat))
    return _func_count_supports_mesh(i_g, d_mesh, collider_info)


@qd.func
def _func_count_supports_mesh(i_g: int, d_mesh: qd.types.vector(3), collider_info: array_class.ColliderInfo):
    """Count the vertices tied for the support of a mesh-frame direction, all of them listed by the cell holding it."""
    i_cell = _func_direction_cell(i_g, d_mesh, collider_info)
    dot_max = collider_info.support_field.support_v[2 * i_cell].dot(d_mesh)
    count = gs.qd_int(1)
    # A cell with a single candidate holds it twice in its record (see SupportFieldInfo)
    if collider_info.support_field.support_vid[2 * i_cell + 1] != collider_info.support_field.support_vid[2 * i_cell]:
        dot = collider_info.support_field.support_v[2 * i_cell + 1].dot(d_mesh)
        if dot > dot_max:
            dot_max = dot
        elif dot == dot_max:
            count += 1
    i_s_start = collider_info.support_field.support_list_start[i_cell]
    i_s_end = collider_info.support_field.support_list_start[i_cell + 1]
    for i_s in range(i_s_start, i_s_end):
        dot = collider_info.support_field.support_v[i_s].dot(d_mesh)
        if dot > dot_max:
            count = 1
            dot_max = dot
        elif dot == dot_max:
            count += 1

    return count


@qd.func
def _func_count_supports_box(d: qd.types.vector(3), quat: qd.types.vector(4)):
    """
    Count the number of valid support points for a box in the given direction.

    Only needs quat since counting doesn't depend on position or geometry-specific data.

    If the direction has 1 zero component, there are 2 possible support points. If the direction has 2 zero
    components, there are 4 possible support points.
    """
    d_box = gu.qd_inv_transform_by_quat(d, quat)

    return 2 ** (d_box == 0.0).cast(gs.qd_int).sum()


from genesis.utils.deprecated_module_wrapper import create_virtual_deprecated_module

create_virtual_deprecated_module(__name__, "genesis.engine.solvers.rigid.support_field_decomp")
