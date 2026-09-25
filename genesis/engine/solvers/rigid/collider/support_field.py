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
    def __init__(self, rigid_solver: "RigidSolver") -> None:
        self.solver = rigid_solver
        self._support_res = 180
        self._support_field_info = array_class.get_support_field_info(0, 0, self._support_res)
        self._is_active = False

    def _get_direction_grid(self):
        """
        Unit direction of every cell of the two spherical charts, as [chart, azimuth, polar].

        A spherical chart cannot resolve the azimuth of a direction along its own polar axis, and which of the tied
        support vertices of such a direction wins then depends on the signs of zeros, hence on how the whole scene is
        oriented. The second chart carries its poles a quarter turn away, so every direction is at least 45 degrees
        off the poles of whichever chart answers for it, and no lookup is ever resolved by a chart degenerate there.
        """
        support_res = self._support_res
        theta = np.arange(support_res) / support_res * 2 * math.pi - math.pi
        phi = np.arange(support_res) / support_res * math.pi

        spherical_coords = np.zeros([support_res, support_res, 2])
        spherical_coords[:, :, 0] = theta[:, None]
        spherical_coords[:, :, 1] = phi[None]

        x = np.sin(spherical_coords[:, :, 1]) * np.cos(spherical_coords[:, :, 0])
        y = np.sin(spherical_coords[:, :, 1]) * np.sin(spherical_coords[:, :, 0])
        z = np.cos(spherical_coords[:, :, 1])
        v = np.stack((x, y, z), axis=-1)

        # The second chart is the first one turned a quarter turn about x, which carries its poles onto +/- y. A chart
        # answers only for the directions it is at least 45 degrees off its own poles for, so only that equatorial
        # band of rows is ever read and only it is stored, one guard row either side for the neighbours the lookup
        # reads. The two bands together hold about as many cells as the single full chart they replace.
        band_start, band_rows = self._polar_band()
        band = slice(band_start, band_start + band_rows)
        return np.stack((v[:, band], np.stack((x, z, -y), axis=-1)[:, band]))

    def _polar_band(self):
        """First stored polar row of a chart, and how many rows it stores."""
        return self._support_res // 4 - 1, self._support_res // 2 + 3

    def activate(self) -> None:
        if self.is_active:
            return

        v = self._get_direction_grid()
        v1 = v.reshape([-1, 3])
        num_v = len(v1)

        support_v = []
        support_vid = []
        support_cell_start = []
        n_support_cells = 0
        if self.solver.n_geoms > 0:
            init_pos = self.solver.dyn_info.verts.init_pos.to_numpy()
            geoms_vert_start = self.solver.dyn_info.geoms.vert_start.to_numpy()
            geoms_vert_end = self.solver.dyn_info.geoms.vert_end.to_numpy()
            for i_g in range(self.solver.n_geoms):
                this_pos = init_pos[geoms_vert_start[i_g] : geoms_vert_end[i_g]]

                window_size = int(5e8 // this_pos.shape[0])
                max_indices = np.empty(num_v, dtype=np.intp)

                for i in range(0, num_v, window_size):
                    end = min(i + window_size, num_v)
                    dot_chunk = v1[i:end] @ this_pos.T
                    max_indices[i:end] = np.argmax(dot_chunk, axis=1)

                support = this_pos[max_indices]

                support_cell_start.append(n_support_cells)
                support_v.append(support)
                support_vid.append(max_indices)
                n_support_cells += support.shape[0]

            support_v = np.concatenate(support_v)
            support_vid = np.concatenate(support_vid, dtype=gs.np_int)
            support_cell_start = np.array(support_cell_start, dtype=gs.np_int)
        else:
            support_v = np.zeros((1, 3), dtype=gs.np_float)
            support_vid = np.zeros((1,), dtype=gs.np_int)
            support_cell_start = np.zeros((1,), dtype=gs.np_int)

        self._support_field_info = array_class.get_support_field_info(
            self.solver.n_geoms, n_support_cells, self._support_res
        )

        _kernel_init_support(
            support_cell_start, support_v, support_vid, self._support_field_info, self.solver.rigid_config
        )

        self._is_active = True

    @property
    def is_active(self):
        return self._is_active


@qd.kernel
def _kernel_init_support(
    support_cell_start: qd.types.ndarray(),
    support_v: qd.types.ndarray(),
    support_vid: qd.types.ndarray(),
    support_field_info: array_class.SupportFieldInfo,
    rigid_config: qd.template(),
):
    n_geoms = support_field_info.support_cell_start.shape[0]
    n_support_cells = support_field_info.support_v.shape[0]

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i in range(n_geoms):
        support_field_info.support_cell_start[i] = support_cell_start[i]

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i in range(n_support_cells):
        support_field_info.support_vid[i] = support_vid[i]
        for j in qd.static(range(3)):
            support_field_info.support_v[i][j] = support_v[i, j]


@qd.func
def _func_support_world(
    i_g,
    d,
    pos: qd.types.vector(3),
    quat: qd.types.vector(4),
    dyn_info: array_class.DynInfo,
    collider_info: array_class.ColliderInfo,
    exhaustive: qd.template() = False,
):
    """Support position for a world direction.

    The vertex comes from the precomputed support table, or from an exhaustive scan of the mesh vertices when
    'exhaustive' is set: the sampled table is approximate and may return a different vertex of a tied flat face,
    while the scan is exact at a cost linear in the vertex count.
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
def _func_direction_grid_coords(d_mesh, support_res):
    """
    Chart of a mesh-frame direction and its continuous cell coordinates within that chart.

    The chart whose polar axis the direction is least aligned with answers, which leaves it at least 45 degrees off
    that chart's poles since the smaller of the two alignments never exceeds one over root two. The azimuth is then
    always well conditioned, so a direction along one of the geom's own axes - the normal of a face resting flat,
    whose support vertices tie - is resolved by a chart that has no degeneracy there, and a geom and any rotated copy
    of it pick the same vertex.
    """
    i_chart = gs.qd_int(0)
    d_chart = d_mesh
    if qd.abs(d_mesh[2]) > qd.abs(d_mesh[1]):
        # The first chart's poles sit on +/- z, so a direction closer to them is handed to the second chart, whose
        # own coordinates see it a quarter turn away.
        i_chart = 1
        d_chart = qd.Vector([d_mesh[0], -d_mesh[2], d_mesh[1]], dt=gs.qd_float)

    theta = qd.atan2(d_chart[1], d_chart[0])  # [-pi, pi]
    phi = qd.acos(d_chart[2])  # [0, pi]

    return i_chart, (theta + math.pi) / math.pi / 2 * support_res, phi / math.pi * support_res


@qd.func
def _func_support_mesh(i_g, d_mesh, collider_info: array_class.ColliderInfo):
    """
    support point at mesh frame coordinate.
    """
    support_res = collider_info.support_field.support_res[None]
    dot_max = gs.qd_float(-1e20)
    v = qd.Vector([0.0, 0.0, 0.0], dt=gs.qd_float)
    vid = 0

    band_start, band_rows = support_res // 4 - 1, support_res // 2 + 3
    i_chart, ii, jj = _func_direction_grid_coords(d_mesh, support_res)
    chart_start = gs.qd_int(collider_info.support_field.support_cell_start[i_g] + i_chart * support_res * band_rows)

    for i4 in range(4):
        i, j = gs.qd_int(0), gs.qd_int(0)
        if i4 % 2:
            i = gs.qd_int(qd.math.ceil(ii)) % support_res
        else:
            i = gs.qd_int(qd.math.floor(ii)) % support_res

        if i4 // 2 > 0:
            j = gs.qd_int(qd.math.clamp(qd.math.ceil(jj) - band_start, 0, band_rows - 1))
        else:
            j = gs.qd_int(qd.math.clamp(qd.math.floor(jj) - band_start, 0, band_rows - 1))

        support_idx = gs.qd_int(chart_start + i * band_rows + j)
        _vid = collider_info.support_field.support_vid[support_idx]
        pos = collider_info.support_field.support_v[support_idx]
        dot = pos.dot(d_mesh)

        if dot > dot_max:
            v = pos
            dot_max = dot
            vid = _vid

    return v, vid


@qd.func
def _func_support_mesh_exhaustive(i_g, d_mesh, dyn_info: array_class.DynInfo):
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
    i_g, d, pos: qd.types.vector(3), quat: qd.types.vector(4), shrink, dyn_info: array_class.DynInfo
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
def _func_support_ellipsoid(i_g, d, pos: qd.types.vector(3), quat: qd.types.vector(4), dyn_info: array_class.DynInfo):
    a = dyn_info.geoms.data[i_g][0]
    b = dyn_info.geoms.data[i_g][1]
    c = dyn_info.geoms.data[i_g][2]

    # Transform direction to ellipsoid local frame
    d_local = gu.qd_inv_transform_by_quat(d, quat)

    # Support point in local frame: diag(a², b², c²) · d_local / ‖diag(a, b, c) · d_local‖
    sx = a * a * d_local[0]
    sy = b * b * d_local[1]
    sz = c * c * d_local[2]
    norm = qd.sqrt(a * a * d_local[0] ** 2 + b * b * d_local[1] ** 2 + c * c * d_local[2] ** 2)
    s_local = qd.Vector([sx / norm, sy / norm, sz / norm], dt=gs.qd_float)

    return pos + gu.qd_transform_by_quat(s_local, quat)


@qd.func
def _func_support_capsule(
    i_g, d, pos: qd.types.vector(3), quat: qd.types.vector(4), shrink, dyn_info: array_class.DynInfo
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
    i_g, d, pos: qd.types.vector(3), quat: qd.types.vector(4), shrink, dyn_info: array_class.DynInfo
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
def _func_support_prism(i_b, d, collider_state: array_class.ColliderState):
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
def _func_support_box(i_g, d, pos: qd.types.vector(3), quat: qd.types.vector(4), dyn_info: array_class.DynInfo):
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
def _func_count_supports_world(i_g, d, quat: qd.types.vector(4), collider_info: array_class.ColliderInfo):
    """
    Count the number of valid support points for the given world direction.
    Only needs quat since counting doesn't depend on position.
    """
    d_mesh = gu.qd_transform_by_quat_fast(d, gu.qd_inv_quat(quat))
    return _func_count_supports_mesh(i_g, d_mesh, collider_info)


@qd.func
def _func_count_supports_mesh(i_g, d_mesh, collider_info: array_class.ColliderInfo):
    """
    Count the number of distinct support vertices tied for the maximum dot product in the given direction.

    The support field is a spherical grid indexed by (theta, phi). We look up 4 neighboring cells using
    floor/ceil of the continuous grid coordinates (ii, jj). When ii or jj is an integer (common for
    axis-aligned directions), floor == ceil and multiple iterations map to the same cell. Without
    deduplication the same vertex is counted multiple times, inflating the count and triggering
    unnecessary perturbation in safe_gjk_support.
    """
    support_res = collider_info.support_field.support_res[None]
    dot_max = gs.qd_float(-1e20)

    band_start, band_rows = support_res // 4 - 1, support_res // 2 + 3
    i_chart, ii, jj = _func_direction_grid_coords(d_mesh, support_res)
    chart_start = gs.qd_int(collider_info.support_field.support_cell_start[i_g] + i_chart * support_res * band_rows)

    # Collect unique cells, deduplicating floor == ceil collisions
    cell_idx = qd.Vector([-1, -1, -1, -1], dt=gs.qd_int)
    cell_dot = qd.Vector([0.0, 0.0, 0.0, 0.0], dt=gs.qd_float)
    n_unique = gs.qd_int(0)

    for i4 in range(4):
        i, j = gs.qd_int(0), gs.qd_int(0)
        if i4 % 2:
            i = gs.qd_int(qd.math.ceil(ii)) % support_res
        else:
            i = gs.qd_int(qd.math.floor(ii)) % support_res

        if i4 // 2 > 0:
            j = gs.qd_int(qd.math.clamp(qd.math.ceil(jj) - band_start, 0, band_rows - 1))
        else:
            j = gs.qd_int(qd.math.clamp(qd.math.floor(jj) - band_start, 0, band_rows - 1))

        support_idx = gs.qd_int(chart_start + i * band_rows + j)

        # Skip duplicate cells (from floor == ceil on integer indices).
        is_dup = False
        for k in range(n_unique):
            if cell_idx[k] == support_idx:
                is_dup = True
        if is_dup:
            continue

        pos = collider_info.support_field.support_v[support_idx]
        cell_dot[n_unique] = pos.dot(d_mesh)
        cell_idx[n_unique] = support_idx
        n_unique += 1

    dot_max = gs.qd_float(-1e20)
    count = gs.qd_int(0)
    for k in range(n_unique):
        if cell_dot[k] > dot_max:
            count = 1
            dot_max = cell_dot[k]
        elif cell_dot[k] == dot_max:
            count += 1

    return count


@qd.func
def _func_count_supports_box(d, quat: qd.types.vector(4)):
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
