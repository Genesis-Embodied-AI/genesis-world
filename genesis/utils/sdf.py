import numpy as np
import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
import genesis.utils.array_class as array_class


class SDF:
    def __init__(self, rigid_solver):
        self.solver = rigid_solver
        self._geoms_sdf_coarse_res = np.array(
            [(geom.sdf_res - 2) // 4 + 1 for geom in rigid_solver.geoms], dtype=gs.np_int
        ).reshape((-1, 3))
        n_coarse_cells = int(self._geoms_sdf_coarse_res.prod(axis=-1).sum())
        self._sdf_info = array_class.get_sdf_info(self.solver.n_geoms, self.solver.n_cells, n_coarse_cells)
        self._is_active = False

    def activate(self):
        if self._is_active:
            return

        if self.solver.n_geoms > 0:
            geoms = self.solver.geoms
            # Coarse min-grid companion of each SDF: block minima over the grid nodes, with blocks overlapping by
            # one node so that every interpolation cell's 8 nodes lie inside the block of its coarse cell. Derived
            # from the loaded grids, so the preprocessing cache is untouched.
            geoms_sdf_coarse_val = []
            for geom, coarse_res in zip(geoms, self._geoms_sdf_coarse_res):
                coarse_val = geom.sdf_val
                for axis in range(3):
                    windows = np.minimum(
                        4 * np.arange(coarse_res[axis])[:, None] + np.arange(5), coarse_val.shape[axis] - 1
                    )
                    coarse_val = np.take(coarse_val, windows, axis=axis).min(axis=axis + 1)
                geoms_sdf_coarse_val.append(coarse_val.reshape((-1,)))
            sdf_kernel_init_geom_fields(
                np.array([geom.cell_start for geom in geoms], dtype=gs.np_int),
                np.concatenate(([0], self._geoms_sdf_coarse_res.prod(axis=-1).cumsum()[:-1]), dtype=gs.np_int),
                np.array([geom.T_mesh_to_sdf for geom in geoms], dtype=gs.np_float),
                np.array([geom.sdf_res for geom in geoms], dtype=gs.np_int),
                np.concatenate([geom.sdf_val_flattened for geom in geoms], dtype=gs.np_float),
                np.concatenate([geom.sdf_grad_flattened for geom in geoms], dtype=gs.np_float),
                np.array([geom.sdf_max for geom in geoms], dtype=gs.np_float),
                np.array(
                    [np.broadcast_to(np.asarray(geom.sdf_cell_size, dtype=gs.np_float), (3,)) for geom in geoms],
                    dtype=gs.np_float,
                ),
                np.concatenate([geom.sdf_closest_vert_flattened for geom in geoms], dtype=gs.np_int),
                self._geoms_sdf_coarse_res,
                np.concatenate(geoms_sdf_coarse_val, dtype=gs.np_float),
                self._sdf_info,
                self.solver.rigid_config,
            )

        self._is_active = True

    @property
    def is_active(self):
        return self._is_active


@qd.kernel
def sdf_kernel_init_geom_fields(
    geoms_sdf_cell_start: qd.types.ndarray(),
    geoms_sdf_coarse_cell_start: qd.types.ndarray(),
    geoms_T_mesh_to_sdf: qd.types.ndarray(),
    geoms_sdf_res: qd.types.ndarray(),
    geoms_sdf_val: qd.types.ndarray(),
    geoms_sdf_grad: qd.types.ndarray(),
    geoms_sdf_max: qd.types.ndarray(),
    geoms_sdf_cell_size: qd.types.ndarray(),
    geoms_sdf_closest_vert: qd.types.ndarray(),
    geoms_sdf_coarse_res: qd.types.ndarray(),
    geoms_sdf_coarse_val: qd.types.ndarray(),
    sdf_info: array_class.SDFInfo,
    rigid_config: qd.template(),
):
    n_geoms = sdf_info.geoms_sdf_start.shape[0]
    n_cells = sdf_info.geoms_sdf_val.shape[0]
    n_coarse_cells = sdf_info.geoms_sdf_coarse_val.shape[0]

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i in range(n_geoms):
        for j, k in qd.static(qd.ndrange(4, 4)):
            sdf_info.geoms_info.T_mesh_to_sdf[i][j, k] = geoms_T_mesh_to_sdf[i, j, k]

        for j in qd.static(range(3)):
            sdf_info.geoms_info.sdf_res[i][j] = geoms_sdf_res[i, j]

        sdf_info.geoms_info.sdf_cell_start[i] = geoms_sdf_cell_start[i]
        sdf_info.geoms_info.sdf_max[i] = geoms_sdf_max[i]
        for j in qd.static(range(3)):
            sdf_info.geoms_info.sdf_cell_size[i][j] = geoms_sdf_cell_size[i, j]
        sdf_info.geoms_info.sdf_coarse_cell_start[i] = geoms_sdf_coarse_cell_start[i]
        for j in qd.static(range(3)):
            sdf_info.geoms_info.sdf_coarse_res[i][j] = geoms_sdf_coarse_res[i, j]

    for i in range(n_cells):
        sdf_info.geoms_sdf_val[i] = geoms_sdf_val[i]
        sdf_info.geoms_sdf_closest_vert[i] = geoms_sdf_closest_vert[i]
        for j in qd.static(range(3)):
            sdf_info.geoms_sdf_grad[i][j] = geoms_sdf_grad[i, j]

    for i in range(n_coarse_cells):
        sdf_info.geoms_sdf_coarse_val[i] = geoms_sdf_coarse_val[i]


@qd.func
def sdf_func_world(
    geom_idx,
    batch_idx,
    pos_world,
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    sdf_info: array_class.SDFInfo,
):
    """
    sdf value from world coordinate
    """

    g_pos = geoms_state.pos[geom_idx, batch_idx]
    g_quat = geoms_state.quat[geom_idx, batch_idx]

    return sdf_func_world_local(geom_idx, pos_world, g_pos, g_quat, geoms_info, sdf_info)


@qd.func
def sdf_func_world_local(
    geom_idx,
    pos_world: qd.types.vector(3),
    geom_pos: qd.types.vector(3),
    geom_quat: qd.types.vector(4),
    geoms_info: array_class.GeomsInfo,
    sdf_info: array_class.SDFInfo,
):
    """
    Computes SDF value from world coordinate, using provided geometry pose
    instead of reading from geoms_state.
    """
    sd = gs.qd_float(0.0)

    if geoms_info.type[geom_idx] == gs.GEOM_TYPE.SPHERE:
        sd = (pos_world - geom_pos).norm() - geoms_info.data[geom_idx][0]

    elif geoms_info.type[geom_idx] == gs.GEOM_TYPE.PLANE:
        pos_mesh = gu.qd_inv_transform_by_trans_quat(pos_world, geom_pos, geom_quat)
        geom_data = geoms_info.data[geom_idx]
        plane_normal = gs.qd_vec3([geom_data[0], geom_data[1], geom_data[2]])
        sd = pos_mesh.dot(plane_normal)

    else:
        pos_mesh = gu.qd_inv_transform_by_trans_quat(pos_world, geom_pos, geom_quat)
        pos_sdf = gu.qd_transform_by_T(pos_mesh, sdf_info.geoms_info.T_mesh_to_sdf[geom_idx])
        sd = sdf_func_sdf(geom_idx, pos_sdf, sdf_info)

    return sd


@qd.func
def sdf_func_coarse_sd_lower_bound(geom_idx, pos_sdf, collider_info: array_class.ColliderInfo):
    """
    Certified lower bound on the trilinear sd at an in-grid point: the minimum node value over the 4^3-cell node
    block containing its interpolation cell. Exact by convexity - the interpolant only combines nodes of that
    block - at the cost of a single load instead of the 8-node gather.
    """
    res = collider_info.sdf.geoms_info.sdf_res[geom_idx]
    base = qd.min(qd.floor(pos_sdf, gs.qd_int), res - 2)
    coarse_cell = base // 4
    coarse_res = collider_info.sdf.geoms_info.sdf_coarse_res[geom_idx]
    return collider_info.sdf.geoms_sdf_coarse_val[
        collider_info.sdf.geoms_info.sdf_coarse_cell_start[geom_idx]
        + (coarse_cell[0] * coarse_res[1] + coarse_cell[1]) * coarse_res[2]
        + coarse_cell[2]
    ]


@qd.func
def sdf_func_world_local_banded(
    geom_idx,
    pos_world: qd.types.vector(3),
    geom_pos: qd.types.vector(3),
    geom_quat: qd.types.vector(4),
    band,
    dyn_info: array_class.DynInfo,
    collider_info: array_class.ColliderInfo,
):
    """
    Band-gated variant of sdf_func_world_local, returning (is_in_band, sd).

    is_in_band == (sd < band) for exactly the sd sdf_func_world_local would return, but sd itself is only computed
    (and meaningful) when in band: an in-grid query whose coarse block minimum already clears the band skips the
    8-node gather for a single coarse load.
    """
    is_in_band = False
    sd = gs.qd_float(0.0)

    if dyn_info.geoms.type[geom_idx] == gs.GEOM_TYPE.SPHERE:
        sd = (pos_world - geom_pos).norm() - dyn_info.geoms.data[geom_idx][0]
        is_in_band = sd < band

    elif dyn_info.geoms.type[geom_idx] == gs.GEOM_TYPE.PLANE:
        pos_mesh = gu.qd_inv_transform_by_trans_quat(pos_world, geom_pos, geom_quat)
        geom_data = dyn_info.geoms.data[geom_idx]
        plane_normal = gs.qd_vec3([geom_data[0], geom_data[1], geom_data[2]])
        sd = pos_mesh.dot(plane_normal)
        is_in_band = sd < band

    else:
        pos_mesh = gu.qd_inv_transform_by_trans_quat(pos_world, geom_pos, geom_quat)
        pos_sdf = gu.qd_transform_by_T(pos_mesh, collider_info.sdf.geoms_info.T_mesh_to_sdf[geom_idx])
        if sdf_func_is_outside_sdf_grid(geom_idx, pos_sdf, collider_info.sdf):
            sd = sdf_func_proxy_sdf(geom_idx, pos_sdf, collider_info.sdf)
            is_in_band = sd < band
        else:
            coarse_lower_bound = sdf_func_coarse_sd_lower_bound(geom_idx, pos_sdf, collider_info)
            # The bound holds in exact arithmetic, but the floating-point evaluation of the trilinear sum can
            # round below the block minimum; the relative guard keeps a vertex whose exact interpolant clears the
            # band from being misclassified when its rounded value dips just inside.
            if not (coarse_lower_bound - 1e-6 * (1.0 + qd.abs(coarse_lower_bound)) >= band):
                sd = sdf_func_true_sdf(geom_idx, pos_sdf, collider_info.sdf)
                is_in_band = sd < band

    return is_in_band, sd


@qd.func
def sdf_func_ray_exit_distance(
    geom_idx,
    origin: qd.types.vector(3),
    direction: qd.types.vector(3),
    max_dist,
    tolerance,
    geom_pos: qd.types.vector(3),
    geom_quat: qd.types.vector(4),
    dyn_info: array_class.DynInfo,
    collider_info: array_class.ColliderInfo,
):
    """
    Distance from a point inside the geom to its surface along a unit direction, bisected down to tolerance.
    """
    dist = max_dist
    sd_end = sdf_func_world_local(
        geom_idx, origin + max_dist * direction, geom_pos, geom_quat, dyn_info.geoms, collider_info.sdf
    )
    if sd_end > 0.0:
        t_lo = gs.qd_float(0.0)
        t_hi = max_dist
        while t_hi - t_lo > tolerance:
            t_mid = 0.5 * (t_lo + t_hi)
            sd_mid = sdf_func_world_local(
                geom_idx, origin + t_mid * direction, geom_pos, geom_quat, dyn_info.geoms, collider_info.sdf
            )
            if sd_mid < 0.0:
                t_lo = t_mid
            else:
                t_hi = t_mid
        dist = 0.5 * (t_lo + t_hi)
    return dist


@qd.func
def sdf_func_sdf(geom_idx, pos_sdf, sdf_info: array_class.SDFInfo):
    """
    sdf value at sdf frame coordinate.
    Note that the stored sdf magnitude is already w.r.t world/mesh frame.
    """
    signed_dist = gs.qd_float(0.0)
    if sdf_func_is_outside_sdf_grid(geom_idx, pos_sdf, sdf_info):
        signed_dist = sdf_func_proxy_sdf(geom_idx, pos_sdf, sdf_info)
    else:
        signed_dist = sdf_func_true_sdf(geom_idx, pos_sdf, sdf_info)
    return signed_dist


@qd.func
def sdf_func_is_outside_sdf_grid(geom_idx, pos_sdf, sdf_info: array_class.SDFInfo):
    res = sdf_info.geoms_info.sdf_res[geom_idx]
    return (pos_sdf >= res - 1).any() or (pos_sdf <= 0).any()


@qd.func
def sdf_func_proxy_sdf(geom_idx, pos_sdf, sdf_info: array_class.SDFInfo):
    """
    Use distance to center as a proxy sdf, strictly greater than any point inside the cube to ensure value comparison
    is valid.

    Only considers region outside of cube. For anisotropic SDF grids the per-axis cell sizes are applied before taking
    the norm so the result remains a world distance.
    """
    center = (sdf_info.geoms_info.sdf_res[geom_idx] - 1) / 2.0
    delta = pos_sdf - center
    cs = sdf_info.geoms_info.sdf_cell_size[geom_idx]
    scaled = qd.Vector([delta[0] * cs[0], delta[1] * cs[1], delta[2] * cs[2]], dt=gs.qd_float)
    return scaled.norm() + sdf_info.geoms_info.sdf_max[geom_idx]


@qd.func
def sdf_func_true_sdf(geom_idx, pos_sdf, sdf_info: array_class.SDFInfo):
    """
    True sdf interpolated using stored sdf grid.
    """
    geom_sdf_res = sdf_info.geoms_info.sdf_res[geom_idx]
    base = qd.min(qd.floor(pos_sdf, gs.qd_int), geom_sdf_res - 2)
    signed_dist = gs.qd_float(0.0)
    for offset in qd.grouped(qd.ndrange(2, 2, 2)):
        pos_cell = base + offset
        w_xyz = 1 - qd.abs(pos_sdf - pos_cell)
        w = w_xyz[0] * w_xyz[1] * w_xyz[2]
        signed_dist = (
            signed_dist
            + w * sdf_info.geoms_sdf_val[sdf_func_ravel_cell_idx(pos_cell, geom_idx, geom_sdf_res, sdf_info)]
        )

    return signed_dist


@qd.func
def sdf_func_ravel_cell_idx(cell_idx, geom_idx, sdf_res, sdf_info: array_class.SDFInfo):
    return (
        sdf_info.geoms_info.sdf_cell_start[geom_idx]
        + cell_idx[0] * sdf_res[1] * sdf_res[2]
        + cell_idx[1] * sdf_res[2]
        + cell_idx[2]
    )


@qd.func
def sdf_func_grad_world(
    geom_idx,
    batch_idx,
    pos_world,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    collider_info: array_class.ColliderInfo,
    collider_static_config: qd.template(),
):
    g_pos = dyn_state.geoms.pos[geom_idx, batch_idx]
    g_quat = dyn_state.geoms.quat[geom_idx, batch_idx]

    return sdf_func_grad_world_local(
        geom_idx, pos_world, g_pos, g_quat, dyn_info.geoms, rigid_info, collider_info.sdf, collider_static_config
    )


@qd.func
def sdf_func_grad(
    geom_idx,
    pos_sdf,
    geoms_info: array_class.GeomsInfo,
    rigid_info: array_class.RigidInfo,
    sdf_info: array_class.SDFInfo,
    collider_static_config: qd.template(),
):
    """
    sdf grad at sdf frame coordinate.

    Note that the stored sdf magnitude is already w.r.t world/mesh frame.
    """
    grad_sdf = qd.Vector.zero(gs.qd_float, 3)
    if sdf_func_is_outside_sdf_grid(geom_idx, pos_sdf, sdf_info):
        grad_sdf = sdf_func_proxy_grad(geom_idx, pos_sdf, rigid_info, sdf_info)
    else:
        grad_sdf = sdf_func_true_grad(geom_idx, pos_sdf, geoms_info, sdf_info, collider_static_config)
    return grad_sdf


@qd.func
def sdf_func_proxy_grad(geom_idx, pos_sdf, rigid_info: array_class.RigidInfo, sdf_info: array_class.SDFInfo):
    """
    Use direction from sdf center, scaled per-axis by the anisotropic cell size, to approximate the gradient
    direction outside the cube.

    The matching :func:`sdf_func_proxy_sdf` distance is `||(pos_sdf - center) * cs||` in world units, whose gradient
    direction (after the chain rule for the diagonal SDF<->mesh transform) is the per-axis-scaled delta. Using the raw
    `pos_sdf - center` would skew outside-grid normals toward fine-resolution axes on anisotropic grids and yield
    directionally wrong contact normals for points falling back on this proxy.
    """
    center = (sdf_info.geoms_info.sdf_res[geom_idx] - 1) / 2.0
    delta = pos_sdf - center
    cs = sdf_info.geoms_info.sdf_cell_size[geom_idx]
    scaled = qd.Vector([delta[0] * cs[0], delta[1] * cs[1], delta[2] * cs[2]], dt=gs.qd_float)
    proxy_sdf_grad = gu.qd_normalize(scaled, rigid_info.EPS[None])
    return proxy_sdf_grad


@qd.func
def sdf_func_true_grad(
    geom_idx,
    pos_sdf,
    geoms_info: array_class.GeomsInfo,
    sdf_info: array_class.SDFInfo,
    collider_static_config: qd.template(),
):
    """
    True sdf grad interpolated using stored sdf grad grid.
    """
    sdf_grad_sdf = qd.Vector.zero(gs.qd_float, 3)
    if geoms_info.type[geom_idx] == gs.GEOM_TYPE.TERRAIN:  # Terrain uses finite difference
        if qd.static(collider_static_config.has_terrain):  # for speed up compilation
            # since we are in sdf frame, delta can be a relatively big value
            delta = gs.qd_float(1e-2)

            for i in qd.static(range(3)):
                inc = pos_sdf
                dec = pos_sdf
                inc[i] += delta
                dec[i] -= delta
                sdf_grad_sdf[i] = (
                    sdf_func_true_sdf(geom_idx, inc, sdf_info) - sdf_func_true_sdf(geom_idx, dec, sdf_info)
                ) / (2 * delta)

    else:
        geom_sdf_res = sdf_info.geoms_info.sdf_res[geom_idx]
        base = qd.min(qd.floor(pos_sdf, gs.qd_int), geom_sdf_res - 2)
        for offset in qd.grouped(qd.ndrange(2, 2, 2)):
            pos_cell = base + offset
            w_xyz = 1 - qd.abs(pos_sdf - pos_cell)
            w = w_xyz[0] * w_xyz[1] * w_xyz[2]
            sdf_grad_sdf = (
                sdf_grad_sdf
                + w * sdf_info.geoms_sdf_grad[sdf_func_ravel_cell_idx(pos_cell, geom_idx, geom_sdf_res, sdf_info)]
            )

    return sdf_grad_sdf


@qd.func
def sdf_func_grad_world_local_consistent(
    geom_idx,
    pos_world: qd.types.vector(3),
    geom_pos: qd.types.vector(3),
    geom_quat: qd.types.vector(4),
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    collider_info: array_class.ColliderInfo,
):
    """
    SDF gradient in world coordinates as the analytic gradient of the trilinear value interpolant, NOT the
    interpolation of precomputed lattice gradients that sdf_func_grad_world_local returns.

    A contact whose normal is the exact gradient of the field supplying its penetration derives from a potential
    and can do no net work around a closed micro-cycle; the lattice-grad interpolation is smoother but tilted off
    the penetration level sets, which makes a settled stack of nested shells ratchet sideways. That smoothness is
    load-bearing for sliding thin features (bolt threads), so only the nested-shell contact path uses this variant.
    """
    EPS = rigid_info.EPS[None]
    grad_world = qd.Vector.zero(gs.qd_float, 3)
    if dyn_info.geoms.type[geom_idx] == gs.GEOM_TYPE.SPHERE:
        grad_world = gu.qd_normalize(pos_world - geom_pos, EPS)
    elif dyn_info.geoms.type[geom_idx] == gs.GEOM_TYPE.PLANE:
        geom_data = dyn_info.geoms.data[geom_idx]
        plane_normal = gs.qd_vec3([geom_data[0], geom_data[1], geom_data[2]])
        grad_world = gu.qd_transform_by_quat(plane_normal, geom_quat)
    else:
        pos_mesh = gu.qd_inv_transform_by_trans_quat(pos_world, geom_pos, geom_quat)
        pos_sdf = gu.qd_transform_by_T(pos_mesh, collider_info.sdf.geoms_info.T_mesh_to_sdf[geom_idx])
        grad_mesh = qd.Vector.zero(gs.qd_float, 3)
        if sdf_func_is_outside_sdf_grid(geom_idx, pos_sdf, collider_info.sdf):
            grad_mesh = sdf_func_proxy_grad(geom_idx, pos_sdf, rigid_info, collider_info.sdf)
        else:
            geom_sdf_res = collider_info.sdf.geoms_info.sdf_res[geom_idx]
            cs = collider_info.sdf.geoms_info.sdf_cell_size[geom_idx]
            base = qd.min(qd.floor(pos_sdf, gs.qd_int), geom_sdf_res - 2)
            for offset in qd.grouped(qd.ndrange(2, 2, 2)):
                pos_cell = base + offset
                w_xyz = 1 - qd.abs(pos_sdf - pos_cell)
                val = collider_info.sdf.geoms_sdf_val[
                    sdf_func_ravel_cell_idx(pos_cell, geom_idx, geom_sdf_res, collider_info.sdf)
                ]
                grad_mesh[0] += (2 * offset[0] - 1) * w_xyz[1] * w_xyz[2] * val / cs[0]
                grad_mesh[1] += w_xyz[0] * (2 * offset[1] - 1) * w_xyz[2] * val / cs[1]
                grad_mesh[2] += w_xyz[0] * w_xyz[1] * (2 * offset[2] - 1) * val / cs[2]
        grad_world = gu.qd_transform_by_quat(grad_mesh, geom_quat)
    return grad_world


@qd.func
def sdf_func_normal_world(
    geom_idx,
    batch_idx,
    pos_world,
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    rigid_info: array_class.RigidInfo,
    sdf_info: array_class.SDFInfo,
    collider_static_config: qd.template(),
):
    g_pos = geoms_state.pos[geom_idx, batch_idx]
    g_quat = geoms_state.quat[geom_idx, batch_idx]

    return sdf_func_normal_world_local(
        geom_idx, pos_world, g_pos, g_quat, geoms_info, rigid_info, sdf_info, collider_static_config
    )


@qd.func
def sdf_func_normal_world_local(
    geom_idx,
    pos_world: qd.types.vector(3),
    geom_pos: qd.types.vector(3),
    geom_quat: qd.types.vector(4),
    geoms_info: array_class.GeomsInfo,
    rigid_info: array_class.RigidInfo,
    sdf_info: array_class.SDFInfo,
    collider_static_config: qd.template(),
):
    """
    Computes normalized SDF gradient (surface normal) in world coordinates,
    using provided geometry pose instead of reading from geoms_state.
    """
    return gu.qd_normalize(
        sdf_func_grad_world_local(
            geom_idx, pos_world, geom_pos, geom_quat, geoms_info, rigid_info, sdf_info, collider_static_config
        ),
        rigid_info.EPS[None],
    )


@qd.func
def sdf_func_grad_world_local(
    geom_idx,
    pos_world: qd.types.vector(3),
    geom_pos: qd.types.vector(3),
    geom_quat: qd.types.vector(4),
    geoms_info: array_class.GeomsInfo,
    rigid_info: array_class.RigidInfo,
    sdf_info: array_class.SDFInfo,
    collider_static_config: qd.template(),
):
    """
    Computes SDF gradient in world coordinates, using provided geometry pose
    instead of reading from geoms_state.
    """
    EPS = rigid_info.EPS[None]

    grad_world = qd.Vector.zero(gs.qd_float, 3)

    if geoms_info.type[geom_idx] == gs.GEOM_TYPE.SPHERE:
        grad_world = gu.qd_normalize(pos_world - geom_pos, EPS)

    elif geoms_info.type[geom_idx] == gs.GEOM_TYPE.PLANE:
        geom_data = geoms_info.data[geom_idx]
        plane_normal = gs.qd_vec3([geom_data[0], geom_data[1], geom_data[2]])
        grad_world = gu.qd_transform_by_quat(plane_normal, geom_quat)

    else:
        pos_mesh = gu.qd_inv_transform_by_trans_quat(pos_world, geom_pos, geom_quat)
        pos_sdf = gu.qd_transform_by_T(pos_mesh, sdf_info.geoms_info.T_mesh_to_sdf[geom_idx])
        grad_sdf = sdf_func_grad(geom_idx, pos_sdf, geoms_info, rigid_info, sdf_info, collider_static_config)

        grad_mesh = grad_sdf  # no rotation between mesh and sdf frame
        grad_world = gu.qd_transform_by_quat(grad_mesh, geom_quat)

    return grad_world


@qd.func
def sdf_func_find_closest_vert(
    geom_idx,
    i_b,
    pos_world,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    collider_info: array_class.ColliderInfo,
):
    """
    Returns vert of geom that's closest to pos_world
    """
    g_pos = dyn_state.geoms.pos[geom_idx, i_b]
    g_quat = dyn_state.geoms.quat[geom_idx, i_b]
    geom_sdf_res = collider_info.sdf.geoms_info.sdf_res[geom_idx]
    pos_mesh = gu.qd_inv_transform_by_trans_quat(pos_world, g_pos, g_quat)
    pos_sdf = gu.qd_transform_by_T(pos_mesh, collider_info.sdf.geoms_info.T_mesh_to_sdf[geom_idx])
    nearest_cell = qd.cast(qd.min(qd.max(pos_sdf, 0), geom_sdf_res - 1), gs.qd_int)
    return (
        collider_info.sdf.geoms_sdf_closest_vert[
            sdf_func_ravel_cell_idx(nearest_cell, geom_idx, geom_sdf_res, collider_info.sdf)
        ]
        + dyn_info.geoms.vert_start[geom_idx]
    )
