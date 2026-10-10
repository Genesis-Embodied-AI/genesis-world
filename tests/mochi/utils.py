import numpy as np
import quadrants as qd

import genesis as gs
from genesis.engine.solvers.mochi.contact_utils import collision_response
from genesis.engine.solvers.mochi.data import (
    FRICTION_MODEL,
    INTEGRATOR,
    LINEAR_TOLERANCE,
    LINESEARCH,
    MochiStaticConfig,
)


def make_static_config(**overrides):
    config = {
        "backend": gs.backend,
        "para_level": gs.PARA_LEVEL.NEVER,
        "integrator": INTEGRATOR.BACKWARD_EULER,
        "use_newton_euler_inertia": False,
        "friction_model": FRICTION_MODEL.C1,
        "linesearch_type": LINESEARCH.RESIDUAL_NORM,
        "linear_tolerance": LINEAR_TOLERANCE.CONSTANT,
        "use_fitted_friction_hessian": True,
        "friction_with_collider_normal": True,
        "fade_friction": True,
        "implicit_normal_force_for_dissipation": False,
        "has_dense": True,
        "use_tiled_cholesky": False,
        "cholesky_tile_size": 16,
        "tiled_n_dofs": 16,
        "has_grid_colliders": False,
        "record_contacts": True,
        "batch_links_info": False,
        "has_soft": False,
        "has_tets": False,
        "has_equalities": False,
        "has_attachments": False,
        "has_pc_colliders": False,
        "has_rod_band": False,
        "has_soft_colliders": False,
        "tet_tree_levels": 0,
    }
    config.update(overrides)
    return MochiStaticConfig(**config)


@qd.kernel
def kernel_plane_collision_response(
    points: qd.types.ndarray(),
    points_stage_start: qd.types.ndarray(),
    normal: qd.types.ndarray(),
    colliding_normal: qd.types.ndarray(),
    params: qd.types.ndarray(),
    dt_stage: float,
    eps: float,
    energy: qd.types.ndarray(),
    force: qd.types.ndarray(),
    dforce: qd.types.ndarray(),
    mochi_config: qd.template(),
):
    """Contact response of sample points against the plane through the origin with the given normal: the signed
    distance is the projection on the normal, so the response is a function of the sample position alone."""
    n = qd.Vector([normal[0], normal[1], normal[2]], dt=gs.qd_float)
    n_colliding = qd.Vector([colliding_normal[0], colliding_normal[1], colliding_normal[2]], dt=gs.qd_float)
    for i_p in range(points.shape[0]):
        p = qd.Vector([points[i_p, 0], points[i_p, 1], points[i_p, 2]], dt=gs.qd_float)
        p_start = qd.Vector(
            [points_stage_start[i_p, 0], points_stage_start[i_p, 1], points_stage_start[i_p, 2]], dt=gs.qd_float
        )
        d = p.dot(n)
        p_rel = p - p_start
        d_start = d - n.dot(p_rel)
        e, f, D, _ = collision_response(
            d,
            n,
            n_colliding,
            p_rel,
            d_start,
            params[0],
            params[1],
            params[2],
            params[3],
            params[4],
            params[5],
            params[6],
            params[7],
            dt_stage,
            eps,
            mochi_config,
        )
        energy[i_p] = e
        for k in qd.static(range(3)):
            force[i_p, k] = f[k]
            for l in qd.static(range(3)):
                dforce[i_p, k, l] = D[k, l]


def plane_collision_response(points, points_stage_start, normal, colliding_normal, params, dt_stage, mochi_config):
    """Energy, force and force derivative (per unit area) of every point against the plane through the origin."""
    points = np.asarray(points, dtype=gs.np_float).reshape((-1, 3))
    n_points = points.shape[0]
    energy = np.zeros((n_points,), dtype=gs.np_float)
    force = np.zeros((n_points, 3), dtype=gs.np_float)
    dforce = np.zeros((n_points, 3, 3), dtype=gs.np_float)
    kernel_plane_collision_response(
        points,
        np.asarray(points_stage_start, dtype=gs.np_float).reshape((-1, 3)),
        np.asarray(normal, dtype=gs.np_float),
        np.asarray(colliding_normal, dtype=gs.np_float),
        np.asarray(params, dtype=gs.np_float),
        float(dt_stage),
        float(gs.EPS),
        energy,
        force,
        dforce,
        mochi_config,
    )
    return energy, force, dforce
