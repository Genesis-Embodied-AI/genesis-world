import numpy as np
import trimesh

import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
from genesis.engine.entities.rigid_entity.description import SignedDistanceFieldDescription
from genesis.engine.solvers.rigid.data import GEOMS, DynInfo, DynState, GeomsInfo, GeomsState, RigidInfo
from genesis.utils.cache import get_hashkey
from genesis.utils.data import Data, DataKind, Size, field, is_opted_in


# The nodes of the signed distance field (SDF) grids and of their coarse min-grids, all rows together
SDF_CELLS = Size("sdf_cells")
SDF_COARSE_CELLS = Size("sdf_coarse_cells")
# The SDF grids of the geoms, followed by those of the meshes the coupled geoms were decomposed from (see 'SDF')
SDF_ROWS = Size("sdf_rows")


class SDFGeomInfo(Data, kind=DataKind.CONSTANT, shape=(SDF_ROWS,)):
    """
    The signed distance field (SDF) grid of each row: its frame, resolution, cell size, maximum distance and first
    node, and the resolution and first cell of its coarse grid, whose block minima bound the signed distance from below.
    """

    T_mesh_to_sdf: qd.Tensor = field((float, (4, 4)))
    sdf_res: qd.Tensor = field((int, 3))
    sdf_max: qd.Tensor = field(float)
    sdf_cell_size: qd.Tensor = field((float, 3))
    sdf_cell_start: qd.Tensor = field(int)
    # Coarse grid of the block minima of the nodes, a lower bound of the trilinear signed distance
    sdf_coarse_res: qd.Tensor = field((int, 3))
    sdf_coarse_cell_start: qd.Tensor = field(int)


class SDFInfo(Data, kind=DataKind.CONSTANT, shape=(SDF_CELLS,)):
    """
    The nodes of the signed distance field (SDF) grids: signed distance, gradient, closest vertex and coarse minima.
    """

    geoms_info: SDFGeomInfo
    geoms_sdf_val: qd.Tensor = field(float)
    geoms_sdf_grad: qd.Tensor = field(
        (float, 3),
        oversize_hint="Consider setting a larger 'sdf_cell_size', or a smaller 'sdf_max_res', in 'gs.materials.Rigid' "
        "options.",
    )
    geoms_sdf_closest_vert: qd.Tensor = field(int)
    geoms_sdf_coarse_val: qd.Tensor = field(float, shape=(SDF_COARSE_CELLS,))
    # The row of the mesh each geom was decomposed from, -1 for a geom kept whole or uncoupled, for the couplings that
    # see the source mesh instead of the union of its pieces
    geoms_source_row: qd.Tensor = field(int, shape=(GEOMS,), opt_in=True)


class SDF:
    """Build the signed distance field (SDF) grids of the rigid geoms once, at their exact size.

    The geoms of one trimesh share one grid whenever their grids have the same resolution, cell size and gradient flag
    (see 'Mesh.tmesh_hash'), as the instances of an asset do. When a coupling opts in 'geoms_source_row', the meshes the
    coupled geoms were decomposed from get rows of their own after those of the geoms, shared the same way. The grids
    are loaded and written into the SDF info at construction.
    """

    def __init__(self, rigid_solver):
        # The grids are costly to load, so they are only for a scene with a consumer opting the SDF info in, the others
        # holding placeholders the kernels never read
        if not is_opted_in(SDFInfo):
            self.sdf_info = SDFInfo(unused=True)
            return

        geoms = rigid_solver.geoms
        has_sources = is_opted_in(SDFInfo.geoms_source_row)
        # The mesh and grid of each row, the geoms first then the source meshes of the coupled decomposed geoms, whose
        # grids are resolved as those of the geoms (see 'SignedDistanceFieldDescription.from_geometry')
        rows_mesh, rows_sdf = [geom.mesh for geom in geoms], [geom.desc.sdf for geom in geoms]
        geoms_source_row = np.full((len(geoms),), -1, dtype=gs.np_int)
        if has_sources:
            sources_row_by_key: dict[tuple[str, int], int] = {}
            for i_g, geom in enumerate(geoms):
                source_verts, source_faces = geom.desc.source_verts, geom.desc.source_faces
                if geom.needs_coup and source_verts is not None:
                    key = (get_hashkey(source_verts, source_faces), id(geom.entity.material))
                    i_row = sources_row_by_key.setdefault(key, len(rows_mesh))
                    if i_row == len(rows_mesh):
                        source_mesh = gs.Mesh.from_trimesh(
                            trimesh.Trimesh(source_verts, source_faces, process=False),
                            surface=gs.surfaces.Collision(),
                        )
                        rows_mesh.append(source_mesh)
                        rows_sdf.append(
                            SignedDistanceFieldDescription.from_geometry(
                                source_mesh, gs.GEOM_TYPE.MESH, geom.entity.material
                            )
                        )
                    geoms_source_row[i_g] = i_row
        rows_grid_idx = np.empty((len(rows_mesh),), dtype=gs.np_int)
        grids_idx_by_key: dict[tuple[str, SignedDistanceFieldDescription], int] = {}
        grids_mesh, grids_desc = [], []
        for i_row, (mesh, sdf) in enumerate(zip(rows_mesh, rows_sdf)):
            key = (mesh.tmesh_hash, sdf)
            i_grid = grids_idx_by_key.setdefault(key, len(grids_desc))
            if i_grid == len(grids_desc):
                grids_mesh.append(mesh)
                grids_desc.append(sdf)
            rows_grid_idx[i_row] = i_grid
        grids_sdf = [mesh.get_sdf_grid(sdf) for mesh, sdf in zip(grids_mesh, grids_desc)]
        grids_n_nodes = np.array([sdf.n_nodes for sdf in grids_desc], dtype=gs.np_int).reshape((-1, 3))
        grids_n_cells = grids_n_nodes.prod(axis=-1, dtype=gs.np_int)
        grids_sdf_coarse_res = (grids_n_nodes - 2) // 4 + 1
        grids_n_coarse_cells = grids_sdf_coarse_res.prod(axis=-1, dtype=gs.np_int)
        self.sdf_info = SDFInfo(
            sizes=dict(
                geoms=len(geoms),
                sdf_rows=len(rows_mesh),
                sdf_cells=grids_n_cells.sum(),
                sdf_coarse_cells=grids_n_coarse_cells.sum(),
            )
        )
        if geoms:
            # Coarse min-grid companion of each SDF: block minima over the grid nodes, with blocks overlapping by one
            # node so that every interpolation cell's 8 nodes lie inside the block of its coarse cell. Derived from the
            # loaded grids, so the preprocessing cache is untouched.
            grids_sdf_coarse_val = []
            for sdf_grid, coarse_res in zip(grids_sdf, grids_sdf_coarse_res):
                coarse_val = sdf_grid.value
                for axis in range(3):
                    windows = np.minimum(
                        4 * np.arange(coarse_res[axis])[:, None] + np.arange(5), coarse_val.shape[axis] - 1
                    )
                    coarse_val = np.take(coarse_val, windows, axis=axis).min(axis=axis + 1)
                grids_sdf_coarse_val.append(coarse_val.reshape((-1,)))
            rows_sdf_grid = [grids_sdf[i_grid] for i_grid in rows_grid_idx]
            sdf_kernel_init_geom_fields(
                (np.cumsum(grids_n_cells, dtype=gs.np_int) - grids_n_cells)[rows_grid_idx],
                (np.cumsum(grids_n_coarse_cells, dtype=gs.np_int) - grids_n_coarse_cells)[rows_grid_idx],
                np.array([sdf_grid.T_mesh_to_sdf for sdf_grid in rows_sdf_grid], dtype=gs.np_float),
                np.array([sdf.n_nodes for sdf in rows_sdf], dtype=gs.np_int),
                np.concatenate([sdf_grid.value.reshape((-1,)) for sdf_grid in grids_sdf], dtype=gs.np_float),
                np.concatenate([sdf_grid.grad.reshape((-1, 3)) for sdf_grid in grids_sdf], dtype=gs.np_float),
                np.array([sdf_grid.max_value for sdf_grid in rows_sdf_grid], dtype=gs.np_float),
                np.array([sdf.cell_size for sdf in rows_sdf], dtype=gs.np_float),
                np.concatenate([sdf_grid.closest_verts_idx.reshape((-1,)) for sdf_grid in grids_sdf], dtype=gs.np_int),
                grids_sdf_coarse_res[rows_grid_idx],
                np.concatenate(grids_sdf_coarse_val, dtype=gs.np_float),
                geoms_source_row,
                self.sdf_info,
                rigid_solver.rigid_config,
                has_sources=has_sources,
            )


@qd.kernel(fastcache=True)
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
    geoms_source_row: qd.types.ndarray(),
    sdf_info: SDFInfo,
    rigid_config: qd.template(),
    has_sources: qd.template(),
):
    """Fill the signed distance field (SDF) info of the rigid geoms, from per-row arrays in the order of the rows."""
    n_cells = sdf_info.geoms_sdf_val.shape[0]
    n_coarse_cells = sdf_info.geoms_sdf_coarse_val.shape[0]

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_g in range(geoms_sdf_cell_start.shape[0]):
        for j, k in qd.static(qd.ndrange(4, 4)):
            sdf_info.geoms_info.T_mesh_to_sdf[i_g][j, k] = geoms_T_mesh_to_sdf[i_g, j, k]

        for j in qd.static(range(3)):
            sdf_info.geoms_info.sdf_res[i_g][j] = geoms_sdf_res[i_g, j]

        sdf_info.geoms_info.sdf_cell_start[i_g] = geoms_sdf_cell_start[i_g]
        sdf_info.geoms_info.sdf_max[i_g] = geoms_sdf_max[i_g]
        for j in qd.static(range(3)):
            sdf_info.geoms_info.sdf_cell_size[i_g][j] = geoms_sdf_cell_size[i_g, j]
        sdf_info.geoms_info.sdf_coarse_cell_start[i_g] = geoms_sdf_coarse_cell_start[i_g]
        for j in qd.static(range(3)):
            sdf_info.geoms_info.sdf_coarse_res[i_g][j] = geoms_sdf_coarse_res[i_g, j]

    for i in range(n_cells):
        sdf_info.geoms_sdf_val[i] = geoms_sdf_val[i]
        sdf_info.geoms_sdf_closest_vert[i] = geoms_sdf_closest_vert[i]
        for j in qd.static(range(3)):
            sdf_info.geoms_sdf_grad[i][j] = geoms_sdf_grad[i, j]

    for i in range(n_coarse_cells):
        sdf_info.geoms_sdf_coarse_val[i] = geoms_sdf_coarse_val[i]

    if qd.static(has_sources):
        for i_g in range(geoms_source_row.shape[0]):
            sdf_info.geoms_source_row[i_g] = geoms_source_row[i_g]


@qd.func
def sdf_func_world(geom_idx, batch_idx, pos_world, geoms_state: GeomsState, geoms_info: GeomsInfo, sdf_info: SDFInfo):
    """
    sdf value from world coordinate
    """
    return sdf_func_world_row(geom_idx, geom_idx, batch_idx, pos_world, geoms_state, geoms_info, sdf_info)


@qd.func
def sdf_func_world_row(
    geom_idx: int,
    sdf_idx: int,
    batch_idx: int,
    pos_world: qd.types.vector(3),
    geoms_state: GeomsState,
    geoms_info: GeomsInfo,
    sdf_info: SDFInfo,
):
    """
    Return the signed distance of a point to a geom, read from a given SDF grid.

    The geom `geom_idx` gives the pose and the type, and the row `sdf_idx` the grid, its own or that of the mesh it was
    decomposed from (see 'SDF').
    """
    g_pos = geoms_state.pos[geom_idx, batch_idx]
    g_quat = geoms_state.quat[geom_idx, batch_idx]

    return sdf_func_world_local_row(geom_idx, sdf_idx, pos_world, g_pos, g_quat, geoms_info, sdf_info)


@qd.func
def sdf_func_world_local(
    geom_idx,
    pos_world: qd.types.vector(3),
    geom_pos: qd.types.vector(3),
    geom_quat: qd.types.vector(4),
    geoms_info: GeomsInfo,
    sdf_info: SDFInfo,
):
    """
    Computes SDF value from world coordinate, using provided geometry pose
    instead of reading from geoms_state.
    """
    return sdf_func_world_local_row(geom_idx, geom_idx, pos_world, geom_pos, geom_quat, geoms_info, sdf_info)


@qd.func
def sdf_func_world_local_row(
    geom_idx: int,
    sdf_idx: int,
    pos_world: qd.types.vector(3),
    geom_pos: qd.types.vector(3),
    geom_quat: qd.types.vector(4),
    geoms_info: GeomsInfo,
    sdf_info: SDFInfo,
):
    """
    Return the signed distance of a point to a geom of the given pose, read from a given SDF grid.

    The geom `geom_idx` gives the type, and the row `sdf_idx` the grid (see 'sdf_func_world_row').
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
        pos_sdf = gu.qd_transform_by_T(pos_mesh, sdf_info.geoms_info.T_mesh_to_sdf[sdf_idx])
        sd = sdf_func_sdf(sdf_idx, pos_sdf, sdf_info)

    return sd


@qd.func
def sdf_func_coarse_sd_lower_bound(geom_idx, pos_sdf, sdf_info: SDFInfo):
    """
    Certified lower bound on the trilinear sd at an in-grid point: the minimum node value over the 4^3-cell node
    block containing its interpolation cell. Exact by convexity - the interpolant only combines nodes of that
    block - at the cost of a single load instead of the 8-node gather.
    """
    res = sdf_info.geoms_info.sdf_res[geom_idx]
    base = qd.min(qd.floor(pos_sdf, gs.qd_int), res - 2)
    coarse_cell = base // 4
    coarse_res = sdf_info.geoms_info.sdf_coarse_res[geom_idx]
    return sdf_info.geoms_sdf_coarse_val[
        sdf_info.geoms_info.sdf_coarse_cell_start[geom_idx]
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
    dyn_info: DynInfo,
    sdf_info: SDFInfo,
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
        pos_sdf = gu.qd_transform_by_T(pos_mesh, sdf_info.geoms_info.T_mesh_to_sdf[geom_idx])
        if sdf_func_is_outside_sdf_grid(geom_idx, pos_sdf, sdf_info):
            sd = sdf_func_proxy_sdf(geom_idx, pos_sdf, sdf_info)
            is_in_band = sd < band
        else:
            coarse_lower_bound = sdf_func_coarse_sd_lower_bound(geom_idx, pos_sdf, sdf_info)
            # The bound holds in exact arithmetic, but the floating-point evaluation of the trilinear sum can
            # round below the block minimum; the relative guard keeps a vertex whose exact interpolant clears the
            # band from being misclassified when its rounded value dips just inside.
            if not (coarse_lower_bound - 1e-6 * (1.0 + qd.abs(coarse_lower_bound)) >= band):
                sd = sdf_func_true_sdf(geom_idx, pos_sdf, sdf_info)
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
    dyn_info: DynInfo,
    sdf_info: SDFInfo,
):
    """
    Distance from a point inside the geom to its surface along a unit direction, bisected down to tolerance.
    """
    dist = max_dist
    sd_end = sdf_func_world_local(
        geom_idx, origin + max_dist * direction, geom_pos, geom_quat, dyn_info.geoms, sdf_info
    )
    if sd_end > 0.0:
        t_lo = gs.qd_float(0.0)
        t_hi = max_dist
        while t_hi - t_lo > tolerance:
            t_mid = 0.5 * (t_lo + t_hi)
            sd_mid = sdf_func_world_local(
                geom_idx, origin + t_mid * direction, geom_pos, geom_quat, dyn_info.geoms, sdf_info
            )
            if sd_mid < 0.0:
                t_lo = t_mid
            else:
                t_hi = t_mid
        dist = 0.5 * (t_lo + t_hi)
    return dist


@qd.func
def sdf_func_sdf(geom_idx, pos_sdf, sdf_info: SDFInfo):
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
def sdf_func_is_outside_sdf_grid(geom_idx, pos_sdf, sdf_info: SDFInfo):
    res = sdf_info.geoms_info.sdf_res[geom_idx]
    # A non-finite position fails every comparison of the inside test, so it counts as outside and indexes no cell
    return not ((pos_sdf > 0).all() and (pos_sdf < res - 1).all())


@qd.func
def sdf_func_proxy_sdf(geom_idx, pos_sdf, sdf_info: SDFInfo):
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
def sdf_func_true_sdf(geom_idx, pos_sdf, sdf_info: SDFInfo):
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
def sdf_func_ravel_cell_idx(cell_idx, geom_idx, sdf_res, sdf_info: SDFInfo):
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
    dyn_state: DynState,
    dyn_info: DynInfo,
    rigid_info: RigidInfo,
    sdf_info: SDFInfo,
    collider_config: qd.template(),
):
    g_pos = dyn_state.geoms.pos[geom_idx, batch_idx]
    g_quat = dyn_state.geoms.quat[geom_idx, batch_idx]

    return sdf_func_grad_world_local(
        geom_idx, pos_world, g_pos, g_quat, dyn_info.geoms, rigid_info, sdf_info, collider_config
    )


@qd.func
def sdf_func_grad(
    geom_idx,
    sdf_idx: int,
    pos_sdf,
    geoms_info: GeomsInfo,
    rigid_info: RigidInfo,
    sdf_info: SDFInfo,
    collider_config: qd.template(),
):
    """
    sdf grad at sdf frame coordinate.

    Note that the stored sdf magnitude is already w.r.t world/mesh frame.
    """
    grad_sdf = qd.Vector.zero(gs.qd_float, 3)
    if sdf_func_is_outside_sdf_grid(sdf_idx, pos_sdf, sdf_info):
        grad_sdf = sdf_func_proxy_grad(sdf_idx, pos_sdf, rigid_info, sdf_info)
    else:
        grad_sdf = sdf_func_true_grad(geom_idx, sdf_idx, pos_sdf, geoms_info, sdf_info, collider_config)
    return grad_sdf


@qd.func
def sdf_func_proxy_grad(geom_idx, pos_sdf, rigid_info: RigidInfo, sdf_info: SDFInfo):
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
    geom_idx, sdf_idx: int, pos_sdf, geoms_info: GeomsInfo, sdf_info: SDFInfo, collider_config: qd.template()
):
    """
    True sdf grad interpolated using stored sdf grad grid.
    """
    sdf_grad_sdf = qd.Vector.zero(gs.qd_float, 3)
    if geoms_info.type[geom_idx] == gs.GEOM_TYPE.TERRAIN:  # Terrain uses finite difference
        if qd.static(collider_config.has_terrain):  # for speed up compilation
            # since we are in sdf frame, delta can be a relatively big value
            delta = gs.qd_float(1e-2)

            for i in qd.static(range(3)):
                inc = pos_sdf
                dec = pos_sdf
                inc[i] += delta
                dec[i] -= delta
                sdf_grad_sdf[i] = (
                    sdf_func_true_sdf(sdf_idx, inc, sdf_info) - sdf_func_true_sdf(sdf_idx, dec, sdf_info)
                ) / (2 * delta)

    else:
        geom_sdf_res = sdf_info.geoms_info.sdf_res[sdf_idx]
        base = qd.min(qd.floor(pos_sdf, gs.qd_int), geom_sdf_res - 2)
        for offset in qd.grouped(qd.ndrange(2, 2, 2)):
            pos_cell = base + offset
            w_xyz = 1 - qd.abs(pos_sdf - pos_cell)
            w = w_xyz[0] * w_xyz[1] * w_xyz[2]
            sdf_grad_sdf = (
                sdf_grad_sdf
                + w * sdf_info.geoms_sdf_grad[sdf_func_ravel_cell_idx(pos_cell, sdf_idx, geom_sdf_res, sdf_info)]
            )

    return sdf_grad_sdf


@qd.func
def sdf_func_grad_world_local_consistent(
    geom_idx,
    pos_world: qd.types.vector(3),
    geom_pos: qd.types.vector(3),
    geom_quat: qd.types.vector(4),
    dyn_info: DynInfo,
    rigid_info: RigidInfo,
    sdf_info: SDFInfo,
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
        pos_sdf = gu.qd_transform_by_T(pos_mesh, sdf_info.geoms_info.T_mesh_to_sdf[geom_idx])
        grad_mesh = qd.Vector.zero(gs.qd_float, 3)
        if sdf_func_is_outside_sdf_grid(geom_idx, pos_sdf, sdf_info):
            grad_mesh = sdf_func_proxy_grad(geom_idx, pos_sdf, rigid_info, sdf_info)
        else:
            geom_sdf_res = sdf_info.geoms_info.sdf_res[geom_idx]
            cs = sdf_info.geoms_info.sdf_cell_size[geom_idx]
            base = qd.min(qd.floor(pos_sdf, gs.qd_int), geom_sdf_res - 2)
            for offset in qd.grouped(qd.ndrange(2, 2, 2)):
                pos_cell = base + offset
                w_xyz = 1 - qd.abs(pos_sdf - pos_cell)
                val = sdf_info.geoms_sdf_val[sdf_func_ravel_cell_idx(pos_cell, geom_idx, geom_sdf_res, sdf_info)]
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
    geoms_state: GeomsState,
    geoms_info: GeomsInfo,
    rigid_info: RigidInfo,
    sdf_info: SDFInfo,
    collider_config: qd.template(),
):
    return sdf_func_normal_world_row(
        geom_idx, geom_idx, batch_idx, pos_world, geoms_state, geoms_info, rigid_info, sdf_info, collider_config
    )


@qd.func
def sdf_func_normal_world_row(
    geom_idx: int,
    sdf_idx: int,
    batch_idx: int,
    pos_world: qd.types.vector(3),
    geoms_state: GeomsState,
    geoms_info: GeomsInfo,
    rigid_info: RigidInfo,
    sdf_info: SDFInfo,
    collider_config: qd.template(),
):
    """
    Return the outward normal of a geom at a point, read from a given SDF grid.

    The geom `geom_idx` gives the pose and the type, and the row `sdf_idx` the grid (see 'sdf_func_world_row').
    """
    g_pos = geoms_state.pos[geom_idx, batch_idx]
    g_quat = geoms_state.quat[geom_idx, batch_idx]
    grad_world = sdf_func_grad_world_local_row(
        geom_idx, sdf_idx, pos_world, g_pos, g_quat, geoms_info, rigid_info, sdf_info, collider_config
    )
    return gu.qd_normalize(grad_world, rigid_info.EPS[None])


@qd.func
def sdf_func_normal_world_local(
    geom_idx,
    pos_world: qd.types.vector(3),
    geom_pos: qd.types.vector(3),
    geom_quat: qd.types.vector(4),
    geoms_info: GeomsInfo,
    rigid_info: RigidInfo,
    sdf_info: SDFInfo,
    collider_config: qd.template(),
):
    """
    Computes normalized SDF gradient (surface normal) in world coordinates,
    using provided geometry pose instead of reading from geoms_state.
    """
    return gu.qd_normalize(
        sdf_func_grad_world_local(
            geom_idx, pos_world, geom_pos, geom_quat, geoms_info, rigid_info, sdf_info, collider_config
        ),
        rigid_info.EPS[None],
    )


@qd.func
def sdf_func_grad_world_local(
    geom_idx,
    pos_world: qd.types.vector(3),
    geom_pos: qd.types.vector(3),
    geom_quat: qd.types.vector(4),
    geoms_info: GeomsInfo,
    rigid_info: RigidInfo,
    sdf_info: SDFInfo,
    collider_config: qd.template(),
):
    """
    Computes SDF gradient in world coordinates, using provided geometry pose
    instead of reading from geoms_state.
    """
    return sdf_func_grad_world_local_row(
        geom_idx, geom_idx, pos_world, geom_pos, geom_quat, geoms_info, rigid_info, sdf_info, collider_config
    )


@qd.func
def sdf_func_grad_world_local_row(
    geom_idx: int,
    sdf_idx: int,
    pos_world: qd.types.vector(3),
    geom_pos: qd.types.vector(3),
    geom_quat: qd.types.vector(4),
    geoms_info: GeomsInfo,
    rigid_info: RigidInfo,
    sdf_info: SDFInfo,
    collider_config: qd.template(),
):
    """
    Return the SDF gradient of a geom of the given pose at a point, read from a given SDF grid.

    The geom `geom_idx` gives the type, and the row `sdf_idx` the grid (see 'sdf_func_world_row').
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
        pos_sdf = gu.qd_transform_by_T(pos_mesh, sdf_info.geoms_info.T_mesh_to_sdf[sdf_idx])
        grad_sdf = sdf_func_grad(geom_idx, sdf_idx, pos_sdf, geoms_info, rigid_info, sdf_info, collider_config)

        grad_mesh = grad_sdf  # no rotation between mesh and sdf frame
        grad_world = gu.qd_transform_by_quat(grad_mesh, geom_quat)

    return grad_world


@qd.func
def sdf_func_find_closest_vert(geom_idx, i_b, pos_world, dyn_state: DynState, dyn_info: DynInfo, sdf_info: SDFInfo):
    """
    Returns vert of geom that's closest to pos_world
    """
    g_pos = dyn_state.geoms.pos[geom_idx, i_b]
    g_quat = dyn_state.geoms.quat[geom_idx, i_b]
    geom_sdf_res = sdf_info.geoms_info.sdf_res[geom_idx]
    pos_mesh = gu.qd_inv_transform_by_trans_quat(pos_world, g_pos, g_quat)
    pos_sdf = gu.qd_transform_by_T(pos_mesh, sdf_info.geoms_info.T_mesh_to_sdf[geom_idx])
    nearest_cell = qd.cast(qd.min(qd.max(pos_sdf, 0), geom_sdf_res - 1), gs.qd_int)
    return (
        sdf_info.geoms_sdf_closest_vert[sdf_func_ravel_cell_idx(nearest_cell, geom_idx, geom_sdf_res, sdf_info)]
        + dyn_info.geoms.vert_start[geom_idx]
    )
