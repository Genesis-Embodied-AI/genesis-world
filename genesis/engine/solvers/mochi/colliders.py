# Portions of this file are derived from Meta Platforms, Inc. and affiliates' "mochi" physics library
# (mochi_core / mochi_physics), licensed under the Apache License, Version 2.0.
# SPDX-License-Identifier: Apache-2.0
"""Signed distance and gradient of a query point against a collider geom, in the geom frame."""

import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
from genesis.utils import array_class
from genesis.utils.sdf import sdf_func_is_outside_sdf_grid, sdf_func_true_sdf, sdf_func_true_sdf_and_grad

from .data import COLLIDER_TYPE, MochiGeomsInfo

# Bound of the gradient norm of a trilinear interpolant of a signed distance field: each partial derivative is an
# interpolant of unit-bounded finite differences, so the gradient norm is at most sqrt(3). The extrapolation of
# query_collider outside the grid keeps it: every clamped axis trades its interpolant partial for a component of one
# unit vector.
GRID_LIPSCHITZ = 1.7320508075688772


@qd.func
def query_collider(
    i_g,
    pos_geom,
    geoms_info: array_class.GeomsInfo,
    mochi_geoms_info: MochiGeomsInfo,
    sdf_info: array_class.SDFInfo,
    mochi_config: qd.template(),
):
    """Signed distance of a point (geom frame) to the collider geom and the gradient of the distance field.

    Returns whether the collider has a field to query, the signed distance and its gradient in the geom frame.
    Analytic colliders return a unit gradient. Inside its grid, a grid collider returns the trilinear interpolant and
    its exact gradient, whose norm is close to but not exactly one. Outside, it returns the upper bound of the distance
    given by the value at the closest grid point plus the distance to it, so that contacts beyond the grid padding
    (small geoms, wide penalty bands) keep a field continuous across the grid boundary. The gradient then points away
    from the grid along the clamped axes.
    """
    is_valid = True
    sd = gs.qd_float(0.0)
    grad = qd.Vector([1.0, 0.0, 0.0], dt=gs.qd_float)
    collider_type = mochi_geoms_info.collider_type[i_g]
    geom_data = geoms_info.data[i_g]

    if collider_type == COLLIDER_TYPE.PLANE:
        normal = gs.qd_vec3([geom_data[0], geom_data[1], geom_data[2]])
        sd = pos_geom.dot(normal)
        grad = normal
    elif collider_type == COLLIDER_TYPE.SPHERE:
        norm = pos_geom.norm()
        sd = norm - geom_data[0]
        if norm > 0.0:
            grad = pos_geom / norm
    elif collider_type == COLLIDER_TYPE.BOX:
        half = 0.5 * gs.qd_vec3([geom_data[0], geom_data[1], geom_data[2]])
        q = qd.abs(pos_geom) - half
        if q.max() <= 0.0:
            # Inside: the distance to the closest face, along the axis of that face.
            i_max = 0
            if q[1] > q[i_max]:
                i_max = 1
            if q[2] > q[i_max]:
                i_max = 2
            sd = q[i_max]
            grad = qd.Vector.zero(gs.qd_float, 3)
            grad[i_max] = 1.0 if pos_geom[i_max] >= 0.0 else -1.0
        else:
            q_pos = qd.max(q, 0.0)
            sd = q_pos.norm()
            grad = q_pos / sd
            for k in qd.static(range(3)):
                if pos_geom[k] < 0.0:
                    grad[k] = -grad[k]
    else:
        if qd.static(mochi_config.has_grid_colliders):
            pos_sdf = gu.qd_transform_by_T(pos_geom, sdf_info.geoms_info.T_mesh_to_sdf[i_g])
            pos_grid_max = sdf_info.geoms_info.sdf_res[i_g] - 1
            pos_grid = qd.min(qd.max(pos_sdf, 0.0), pos_grid_max)
            sd, grad = sdf_func_true_sdf_and_grad(i_g, pos_grid, sdf_info)
            if sdf_func_is_outside_sdf_grid(i_g, pos_sdf, sdf_info):
                # The SDF frame is a scaled translation of the mesh frame.
                offset = (pos_sdf - pos_grid) * sdf_info.geoms_info.sdf_cell_size[i_g]
                offset_norm = offset.norm()
                sd += offset_norm
                for k in qd.static(range(3)):
                    if (pos_sdf[k] < 0.0 or pos_sdf[k] > pos_grid_max[k]) and offset_norm > 0.0:
                        grad[k] = offset[k] / offset_norm
        else:
            is_valid = False

    return is_valid, sd, grad


@qd.func
def query_collider_lower_bound(
    i_g,
    center_geom,
    radius,
    geoms_info: array_class.GeomsInfo,
    mochi_geoms_info: MochiGeomsInfo,
    sdf_info: array_class.SDFInfo,
    mochi_config: qd.template(),
):
    """Lower bound of the signed distance to the collider over a sphere (geom frame), used to prune whole nodes of a
    sample hierarchy at once.

    The analytic colliders are exact distance fields (1-Lipschitz). The grid field of query_collider is GRID_LIPSCHITZ-
    Lipschitz inside and outside its grid. Outside, it is at least the distance to the grid, which is 1-Lipschitz and
    much cheaper, since the grid padding puts the grid boundary outside the surface."""
    lower = -gs.qd_float(1e30)
    if mochi_geoms_info.collider_type[i_g] == COLLIDER_TYPE.GRID:
        if qd.static(mochi_config.has_grid_colliders):
            pos_sdf = gu.qd_transform_by_T(center_geom, sdf_info.geoms_info.T_mesh_to_sdf[i_g])
            if sdf_func_is_outside_sdf_grid(i_g, pos_sdf, sdf_info):
                pos_grid = qd.min(qd.max(pos_sdf, 0.0), sdf_info.geoms_info.sdf_res[i_g] - 1)
                lower = ((pos_sdf - pos_grid) * sdf_info.geoms_info.sdf_cell_size[i_g]).norm() - radius
            else:
                lower = sdf_func_true_sdf(i_g, pos_sdf, sdf_info) - GRID_LIPSCHITZ * radius
        else:
            lower = gs.qd_float(1e30)
    else:
        _is_valid, sd, _grad = query_collider(i_g, center_geom, geoms_info, mochi_geoms_info, sdf_info, mochi_config)
        lower = sd - radius
    return lower
