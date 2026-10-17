# Portions of this file are derived from Meta Platforms, Inc. and affiliates' "mochi" physics library
# (mochi_core / mochi_physics), licensed under the Apache License, Version 2.0.
# SPDX-License-Identifier: Apache-2.0
"""Forward kinematics of a Newton iterate in one kernel: link poses, joint motion subspaces (the link Jacobians of the
residual), geom poses and geom bounds. The joint-space velocities are left out; the residual never reads them and the
step recomputes them once its final iterate is known."""

import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
from genesis.engine.solvers.rigid.abd.forward_kinematics import (
    func_COM_root,
    func_forward_kinematics_root,
    func_forward_velocity_root,
    func_update_geom_aabbs,
    func_update_geoms_root,
)
from genesis.utils import array_class


@qd.func
def func_forward_kinematics_env(
    i_b: int,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    with_velocity: qd.template(),
):
    """Compute the link poses, the centers of mass and the geom poses of env i_b, one kinematic root after another.

    The joint-space velocities follow the centers of mass under with_velocity.
    """
    for i_r in range(rigid_info.roots_link_idx.shape[0]):
        i_l_root = rigid_info.roots_link_idx[i_r]
        func_forward_kinematics_root(
            i_l_root, i_b, rigid_info.qpos, dyn_state, dyn_info, rigid_info, rigid_config, is_backward=False
        )
        func_COM_root(i_l_root, i_b, dyn_state, dyn_info, rigid_info, rigid_config)
        if qd.static(with_velocity):
            func_forward_velocity_root(i_l_root, i_b, dyn_state, dyn_info, rigid_info, rigid_config, is_backward=False)
        func_update_geoms_root(
            i_l_root,
            i_b,
            dyn_state,
            dyn_info,
            rigid_info,
            rigid_config,
            force_update_all_geoms=False,
            is_backward=False,
        )


@qd.kernel(fastcache=True)
def kernel_update_geom_aabbs(
    geoms_init_AABB: array_class.GeomsInitAABB,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_config: qd.template(),
):
    func_update_geom_aabbs(geoms_init_AABB, dyn_state, dyn_info, rigid_config)


@qd.kernel
def kernel_update_kinematics(
    envs_idx: qd.types.ndarray(),
    geoms_init_AABB: array_class.GeomsInitAABB,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    for i_b_ in range(envs_idx.shape[0]):
        i_b = qd.cast(envs_idx[i_b_], qd.i32)
        func_forward_kinematics_env(i_b, dyn_state, dyn_info, rigid_info, rigid_config, with_velocity=False)
    func_update_geom_aabbs(geoms_init_AABB, dyn_state, dyn_info, rigid_config)


@qd.func
def func_update_kinematics(
    envs: qd.types.ndarray(),
    n_envs: qd.types.ndarray(),
    geoms_init_AABB: array_class.GeomsInitAABB,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    with_velocity: qd.template(),
):
    """Link poses, motion subspaces, geom poses and bounds of the environments of a list (joint-space velocities on
    request); one thread per environment for the kinematic chain, one per geom for the bounds."""
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.ALL))
    for i_slot in range(n_envs[None]):
        i_b = envs[i_slot]
        func_forward_kinematics_env(i_b, dyn_state, dyn_info, rigid_info, rigid_config, with_velocity)
    n_geoms = dyn_state.geoms.pos.shape[0]
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_g, i_slot in qd.ndrange(n_geoms, n_envs[None]):
        i_b = envs[i_slot]
        g_pos = dyn_state.geoms.pos[i_g, i_b]
        g_quat = dyn_state.geoms.quat[i_g, i_b]
        lower = gu.qd_vec3(qd.math.inf)
        upper = gu.qd_vec3(-qd.math.inf)
        for i_corner in qd.static(range(8)):
            corner_pos = gu.qd_transform_by_trans_quat(geoms_init_AABB[i_g, i_corner], g_pos, g_quat)
            lower = qd.min(lower, corner_pos)
            upper = qd.max(upper, corner_pos)
        dyn_state.geoms.aabb_min[i_g, i_b] = lower
        dyn_state.geoms.aabb_max[i_g, i_b] = upper
