import math
from typing import TYPE_CHECKING

import numpy as np

import quadrants as qd

import genesis as gs
import genesis.utils.array_class as array_class
import genesis.utils.geom as gu


if TYPE_CHECKING:
    from genesis.engine.solvers.rigid.rigid_solver import RigidSolver


class SupportField:
    """
    Support table of the geoms: for every cell of a grid of directions, the vertices supporting a direction of the cell.

    A lookup scans the list of the cell its direction falls in, which makes it exact up to the angle error of the
    binning in single precision (see _func_direction_cell). The normal cone of a vertex (the directions it supports)
    meets a cell only by holding one of its corners, by holding the normal of one of its faces inside the cell, or by
    crossing the cell with one of its edges, which are the arcs between the normals of two adjacent hull faces. A cell
    therefore lists the supports of its four corners, the vertices of the hull faces whose normal it holds, and the two
    vertices of every hull edge whose arc crosses it.

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
        # Allocated once, when activated or else empty when first read (see 'support_field_info')
        self._support_field_info: array_class.SupportFieldInfo | None = None
        self._is_active = False

    def activate(self) -> None:
        if self.is_active:
            return

        band_rows = self._support_res // 2
        n_geom_cells = 2 * self._support_res * band_rows

        # The table of a geom depends on its trimesh alone, so the geoms of one trimesh share one table (see
        # 'Mesh.tmesh_hash'), as the instances of an asset do.
        geoms_table_idx = np.zeros((self.solver.n_geoms,), dtype=gs.np_int)
        tables_idx_by_hash: dict[str, int] = {}
        tables_geom = []
        for geom in self.solver.geoms:
            # The support of a terrain is read off the prism built from its height field, so its table would never be
            # read. Its vertex count (one per height field sample) would make it the costliest table to build.
            if geom.type == gs.GEOM_TYPE.TERRAIN:
                continue
            i_t = tables_idx_by_hash.setdefault(geom.mesh.tmesh_hash, len(tables_geom))
            if i_t == len(tables_geom):
                tables_geom.append(geom)
            geoms_table_idx[geom.idx] = i_t
        n_support_cells = n_geom_cells * len(tables_geom)
        # The records of the cells and the extra candidate counts are written in place as the tables are built, the
        # extra candidates alone being collected per table, since their count is unknown beforehand
        records_vid = np.empty((n_support_cells, 2), dtype=gs.np_int)
        support_list_start = np.empty((n_support_cells + 1,), dtype=gs.np_int)
        extras_vid = []
        for i_t, geom in enumerate(tables_geom):
            cell_start = i_t * n_geom_cells

            slots_vid, table_extras_vid, cells_n_extras = geom.mesh.get_support_table(self._support_res)
            records_vid[cell_start : cell_start + n_geom_cells] = slots_vid
            extras_vid.append(table_extras_vid)
            support_list_start[cell_start + 1 : cell_start + n_geom_cells + 1] = cells_n_extras

        # The extra candidates of every cell start past the records of all cells, in cell order
        support_list_start[0] = 2 * n_support_cells
        np.cumsum(support_list_start, out=support_list_start)
        extras_vid = np.concatenate((np.zeros((0,), dtype=gs.np_int), *extras_vid), dtype=gs.np_int)

        assert self._support_field_info is None
        self._support_field_info = array_class.get_support_field_info(
            self.solver.n_geoms, n_support_cells, 2 * n_support_cells + len(extras_vid), self._support_res
        )
        _kernel_init_support(
            n_geom_cells * geoms_table_idx,
            support_list_start,
            np.array([geom.vert_start for geom in tables_geom], dtype=gs.np_int),
            records_vid,
            extras_vid,
            self.solver.dyn_info,
            self._support_field_info,
            self.solver.rigid_config,
            n_geom_cells,
        )

        self._is_active = True

    @property
    def is_active(self):
        return self._is_active

    @property
    def support_field_info(self) -> array_class.SupportFieldInfo:
        """Support tables of the geoms, empty if the field was never activated."""
        if self._support_field_info is None:
            self._support_field_info = array_class.get_support_field_info(0, 0, 0, self._support_res)
        return self._support_field_info


@qd.kernel
def _kernel_init_support(
    support_cell_start: qd.types.ndarray(),
    support_list_start: qd.types.ndarray(),
    tables_vert_start: qd.types.ndarray(),
    records_vid: qd.types.ndarray(),
    extras_vid: qd.types.ndarray(),
    dyn_info: array_class.DynInfo,
    support_field_info: array_class.SupportFieldInfo,
    rigid_config: qd.template(),
    n_geom_cells: int,
):
    """Fill the support table from the vertex indices of its records and extra candidates, along with their positions.

    The records of the cells come first, two per cell, then the extra candidates of all cells (see SupportFieldInfo).
    Every table spans 'n_geom_cells' cells. Its vertex indices are local to its geoms, which all have the same
    vertices, so their positions are read from the first of them, which starts at 'tables_vert_start'.
    """
    n_records = records_vid.shape[0]

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_g in range(support_cell_start.shape[0]):
        support_field_info.support_cell_start[i_g] = support_cell_start[i_g]

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_cell in range(support_list_start.shape[0]):
        support_field_info.support_list_start[i_cell] = support_list_start[i_cell]

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_cell in range(n_records):
        i_v_start = tables_vert_start[i_cell // n_geom_cells]
        for i_k in qd.static(range(2)):
            i_v = records_vid[i_cell, i_k]
            support_field_info.support_vid[2 * i_cell + i_k] = i_v
            support_field_info.support_v[2 * i_cell + i_k] = dyn_info.verts.init_pos[i_v_start + i_v]

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_t in range(tables_vert_start.shape[0]):
        i_v_start = tables_vert_start[i_t]
        i_s_start = support_list_start[i_t * n_geom_cells]
        i_s_end = support_list_start[(i_t + 1) * n_geom_cells]
        for i_s in range(i_s_start, i_s_end):
            i_v = extras_vid[i_s - 2 * n_records]
            support_field_info.support_vid[i_s] = i_v
            support_field_info.support_v[i_s] = dyn_info.verts.init_pos[i_v_start + i_v]


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
    'exhaustive' is set. The scan is exact and returns the first of tied vertices at a cost linear in the vertex count,
    while the table is exact up to the angle error of its binning in single precision (see _func_direction_cell) and
    returns any of them.
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
        v_, vid = _func_support_mesh(i_g, d_mesh, dyn_info, collider_info)
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
        # own coordinates see it a quarter turn away (see CHARTS_ROT in genesis/engine/mesh.py).
        i_chart = 1
        d_chart = qd.Vector([d_mesh[0], -d_mesh[2], d_mesh[1]], dt=gs.qd_float)

    # Azimuth theta in [-pi, pi] and polar angle phi in [0, pi]. In single precision, polynomials spare the libm calls.
    # A direction within their error of a cell boundary may then fall in the neighbouring cell, whose best candidate
    # falls short of the support by that error times the extent of the geom, which stays near the rounding of the dtype
    # in single precision alone.
    theta = 0.0
    phi = 0.0
    if qd.static(qd.lang.impl.get_runtime().default_fp == qd.f32):
        theta = gu.qd_atan2_fast(d_chart[1], d_chart[0])
        phi = gu.qd_acos_fast(d_chart[2])
    else:
        theta = qd.atan2(d_chart[1], d_chart[0])
        phi = qd.acos(d_chart[2])
    # The azimuth wraps around once at most, which a subtraction undoes at the cost of a comparison
    i = gs.qd_int(qd.math.floor((theta + math.pi) / math.pi / 2 * support_res))
    if i >= support_res:
        i = i - support_res
    j = gs.qd_int(qd.math.clamp(qd.math.floor(phi / math.pi * support_res) - band_start, 0, band_rows - 1))
    return collider_info.support_field.support_cell_start[i_g] + (i_chart * support_res + i) * band_rows + j


@qd.func
def _func_support_mesh(
    i_g: int, d_mesh: qd.types.vector(3), dyn_info: array_class.DynInfo, collider_info: array_class.ColliderInfo
):
    """Support vertex of a mesh-frame direction, the best of the candidates listed by the cell it falls in."""
    i_v_start = dyn_info.geoms.vert_start[i_g]
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

    # The table numbers the vertices of its geom from zero (see SupportFieldInfo)
    return v, i_v_start + vid


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
    With 'shrink', the support is the cap centre for any d.
    """
    radius = dyn_info.geoms.data[i_g][0]
    halflength = 0.5 * dyn_info.geoms.data[i_g][1]
    # The point is built in the frame of the cylinder, where the radial part of d has no axial component. Removing the
    # axial component in world frame leaves its rounding, which the division by the radial norm amplifies as d aligns
    # with the axis, lifting the point off the cap by the radius times the ratio of that rounding to the radial part.
    d_local = gu.qd_inv_transform_by_quat(d, quat)
    v_ = qd.Vector([0.0, 0.0, -halflength if d_local[2] < 0.0 else halflength], dt=gs.qd_float)
    if not shrink:
        d_radial_norm = qd.sqrt(d_local[0] ** 2 + d_local[1] ** 2)
        if d_radial_norm > 0.0:
            v_[0] = radius * d_local[0] / d_radial_norm
            v_[1] = radius * d_local[1] / d_radial_norm
    return gu.qd_transform_by_trans_quat_fast(v_, pos, quat)


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
