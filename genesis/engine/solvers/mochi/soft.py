# Portions of this file are derived from Meta Platforms, Inc. and affiliates' "mochi" physics library
# (mochi_core / mochi_physics), licensed under the Apache License, Version 2.0.
# SPDX-License-Identifier: Apache-2.0
"""Deformable bodies: linear tetrahedra whose vertex positions are unknowns of the implicit solve.

Per tetrahedron the incremental potential collects the inertia with the consistent (4-point quadrature) mass matrix,
gravity, mass-proportional damping, the elastic energy with a positive-semidefinite projected tangent, and the
Kelvin-Voigt stiffness damping; the 12x12 tangent block is stored per element and applied on the fly by the linear
solvers. Contact acts on quadrature samples of the boundary triangles against the rigid colliders, scattering to the
three vertices with the barycentric weights; the rigid side reuses the per-pair accumulators of the rigid contact.
"""

import numpy as np
import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
from genesis.utils import array_class

from .articulated import func_jacobian_times_dofs, func_jacobian_transpose_add, func_link_dof_jacobian
from .colliders import query_collider
from .contact import CONSERVATIVE_MAX_ACCEL, CONSERVATIVE_SPEED_SCALE
from .contact_utils import collision_response, func_mat3_to_sym6, func_sym6_to_mat3
from .data import (
    COLLIDER_TYPE,
    INTEGRATOR,
    N_HISTORY,
    SOLVE_STATUS,
    MochiContactState,
    MochiHitReadback,
    MochiInfo,
    MochiSoftInfo,
    MochiSoftState,
    MochiState,
)
from .integration import BDF2_ALPHA_2
from .islands import MochiIslandState
from .lie import skew
from .newton import func_is_env_active
from .rod import (
    func_rod_axial,
    func_rod_axial_strain,
    func_rod_bend_twist,
    func_rod_bend_twist_measures,
    func_rod_normalize,
    func_rod_transport_axis,
)
from .rod_solver import func_rod_band_solve
from .shell import func_shell_elastic, func_shell_rest_data, func_shell_strains
from .soft_materials import (
    func_elastic_energy,
    func_elastic_stress,
    func_stiffness_damping_block,
    func_stiffness_damping_stress,
    func_tet_stiffness,
)

# Kinds of deformable entities.
SOFT_KIND_SOLID = 0
SOFT_KIND_SHELL = 1
SOFT_KIND_ROD = 2
# Guard of the shell geometry (degenerate metrics, normals and averaged normals).
SHELL_TINY = 1e-30

# Consistent mass matrix of a linear tetrahedron, in units of rho V / 20.
CONSISTENT_MASS = ((2.0, 1.0, 1.0, 1.0), (1.0, 2.0, 1.0, 1.0), (1.0, 1.0, 2.0, 1.0), (1.0, 1.0, 1.0, 2.0))

# Columns of the per-entity parameter table passed at initialization.
ENTITY_PARAMS = (
    "mass",
    "has_gravity",
    "model",
    "mu",
    "lam",
    "rho",
    "mass_damping",
    "stiffness_damping",
    "penalty_coefficient",
    "penalty_smoothing_half_distance",
    "penalty_threshold",
    "friction",
    "friction_falloff_vel",
    "viscous_friction",
    "normal_viscous_damping",
    "max_alignment_normals",
    "vert_start",
    "vert_end",
    "sample_start",
    "sample_end",
    "collider_type",
    "sdf_start",
    "sdf_res_x",
    "sdf_res_y",
    "sdf_res_z",
    "sdf_origin_x",
    "sdf_origin_y",
    "sdf_origin_z",
    "sdf_cell_x",
    "sdf_cell_y",
    "sdf_cell_z",
    "kind",
    "membrane_mu",
    "membrane_lambda",
    "bending_alpha",
    "bending_beta",
    "collider_radius",
    "axial_stiffness",
    "torsional_stiffness",
    "rot_inertia",
    "self_contact",
    "self_contact_exclusion_ratio",
)


def build_soft_samples(verts, surface_tri, quadrature_bary, quadrature_weights):
    """Boundary contact samples of a tetrahedral mesh: for each boundary triangle, the quadrature points as barycentric
    coordinates with their rest area weights. Returns (triangles, barycentric coordinates, weights)."""
    tri = verts[surface_tri]
    areas_2 = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=-1)
    n_q = len(quadrature_weights)
    triangles = np.repeat(surface_tri, n_q, axis=0)
    bary = np.tile(quadrature_bary, (len(surface_tri), 1))
    weights = (np.asarray(quadrature_weights)[None, :] * areas_2[:, None]).reshape((-1,))
    return triangles.astype(gs.np_int), bary.astype(gs.np_float), weights.astype(gs.np_float)


# ------------------------------------------------------------------------------------
# ----------------------------------- initialization ---------------------------------
# ------------------------------------------------------------------------------------


@qd.kernel
def kernel_init_soft_fields(
    verts_rest: qd.types.ndarray(),
    verts_entity_idx: qd.types.ndarray(),
    elems_v: qd.types.ndarray(),
    elems_entity_idx: qd.types.ndarray(),
    samples_tri: qd.types.ndarray(),
    samples_bary: qd.types.ndarray(),
    samples_weight: qd.types.ndarray(),
    samples_entity_idx: qd.types.ndarray(),
    entities_params: qd.types.ndarray(),
    entities_links_pair_enabled: qd.types.ndarray(),
    entities_pair_enabled: qd.types.ndarray(),
    sdf_values: qd.types.ndarray(),
    soft_info: MochiSoftInfo,
    soft_state: MochiSoftState,
    rigid_config: qd.template(),
):
    n_verts = verts_rest.shape[0]
    n_elems = elems_v.shape[0]
    n_samples = samples_tri.shape[0]
    n_entities = entities_params.shape[0]
    n_links = entities_links_pair_enabled.shape[1]
    _B = soft_state.verts_pos.shape[1]

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_v in range(n_verts):
        for k in qd.static(range(3)):
            soft_info.verts_rest[i_v][k] = verts_rest[i_v, k]
        soft_info.verts_entity_idx[i_v] = verts_entity_idx[i_v]
        soft_info.verts_mass[i_v] = 0.0
        soft_info.verts_collider_weight[i_v] = 0.0

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_e in range(n_entities):
        soft_info.entities_mass[i_e] = entities_params[i_e, 0]
        soft_info.entities_has_gravity[i_e] = entities_params[i_e, 1] > 0.5
        soft_info.entities_model[i_e] = qd.cast(entities_params[i_e, 2], gs.qd_int)
        soft_info.entities_mu[i_e] = entities_params[i_e, 3]
        soft_info.entities_lam[i_e] = entities_params[i_e, 4]
        soft_info.entities_rho[i_e] = entities_params[i_e, 5]
        soft_info.entities_mass_damping[i_e] = entities_params[i_e, 6]
        soft_info.entities_stiffness_damping[i_e] = entities_params[i_e, 7]
        soft_info.entities_penalty_coefficient[i_e] = entities_params[i_e, 8]
        soft_info.entities_penalty_smoothing_half_distance[i_e] = entities_params[i_e, 9]
        soft_info.entities_penalty_threshold[i_e] = entities_params[i_e, 10]
        soft_info.entities_friction[i_e] = entities_params[i_e, 11]
        soft_info.entities_friction_falloff_vel[i_e] = entities_params[i_e, 12]
        soft_info.entities_viscous_friction[i_e] = entities_params[i_e, 13]
        soft_info.entities_normal_viscous_damping[i_e] = entities_params[i_e, 14]
        soft_info.entities_max_alignment_normals[i_e] = entities_params[i_e, 15]
        soft_info.entities_vert_start[i_e] = qd.cast(entities_params[i_e, 16], gs.qd_int)
        soft_info.entities_vert_end[i_e] = qd.cast(entities_params[i_e, 17], gs.qd_int)
        soft_info.entities_sample_start[i_e] = qd.cast(entities_params[i_e, 18], gs.qd_int)
        soft_info.entities_sample_end[i_e] = qd.cast(entities_params[i_e, 19], gs.qd_int)
        soft_info.entities_collider_type[i_e] = qd.cast(entities_params[i_e, 20], gs.qd_int)
        soft_info.entities_sdf_start[i_e] = qd.cast(entities_params[i_e, 21], gs.qd_int)
        for k in qd.static(range(3)):
            soft_info.entities_sdf_res[i_e][k] = qd.cast(entities_params[i_e, 22 + k], gs.qd_int)
            soft_info.entities_sdf_origin[i_e][k] = entities_params[i_e, 25 + k]
            soft_info.entities_sdf_cell[i_e][k] = entities_params[i_e, 28 + k]
        soft_info.entities_kind[i_e] = qd.cast(entities_params[i_e, 31], gs.qd_int)
        soft_info.entities_membrane_mu[i_e] = entities_params[i_e, 32]
        soft_info.entities_membrane_lambda[i_e] = entities_params[i_e, 33]
        soft_info.entities_bending_alpha[i_e] = entities_params[i_e, 34]
        soft_info.entities_bending_beta[i_e] = entities_params[i_e, 35]
        soft_info.entities_collider_radius[i_e] = entities_params[i_e, 36]
        soft_info.entities_axial_stiffness[i_e] = entities_params[i_e, 37]
        soft_info.entities_torsional_stiffness[i_e] = entities_params[i_e, 38]
        soft_info.entities_rot_inertia[i_e] = entities_params[i_e, 39]
        soft_info.entities_self_contact[i_e] = qd.cast(entities_params[i_e, 40], gs.qd_int)
        soft_info.entities_self_contact_exclusion_ratio[i_e] = entities_params[i_e, 41]
        for i_l in range(n_links):
            soft_info.entities_links_pair_enabled[i_e, i_l] = entities_links_pair_enabled[i_e, i_l]
        for j_e in range(n_entities):
            soft_info.entities_pair_enabled[i_e, j_e] = entities_pair_enabled[i_e, j_e]

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_x in range(sdf_values.shape[0]):
        soft_info.sdf_values[i_x] = sdf_values[i_x]

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_el in range(n_elems):
        for k in qd.static(range(4)):
            soft_info.elems_v[i_el][k] = elems_v[i_el, k]
        soft_info.elems_entity_idx[i_el] = elems_entity_idx[i_el]
        x3 = soft_info.verts_rest[elems_v[i_el, 3]]
        Dm = qd.Matrix.cols(
            [
                soft_info.verts_rest[elems_v[i_el, 0]] - x3,
                soft_info.verts_rest[elems_v[i_el, 1]] - x3,
                soft_info.verts_rest[elems_v[i_el, 2]] - x3,
            ]
        )
        vol = Dm.determinant() / 6.0
        soft_info.elems_Dm[i_el] = Dm
        soft_info.elems_Dm_inv[i_el] = Dm.inverse()
        soft_info.elems_vol[i_el] = vol
        rho = soft_info.entities_rho[elems_entity_idx[i_el]]
        for k in qd.static(range(4)):
            qd.atomic_add(soft_info.verts_mass[elems_v[i_el, k]], 0.25 * rho * vol)

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_s in range(n_samples):
        for k in qd.static(range(3)):
            soft_info.samples_tri[i_s][k] = samples_tri[i_s, k]
            soft_info.samples_bary[i_s][k] = samples_bary[i_s, k]
        soft_info.samples_weight[i_s] = samples_weight[i_s]
        soft_info.samples_entity_idx[i_s] = samples_entity_idx[i_s]

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_v, i_b in qd.ndrange(n_verts, _B):
        pos = soft_info.verts_rest[i_v]
        soft_state.verts_pos[i_v, i_b] = pos
        soft_state.verts_vel[i_v, i_b] = qd.Vector.zero(gs.qd_float, 3)
        for k in qd.static(range(N_HISTORY)):
            soft_state.verts_pos_prev[k, i_v, i_b] = pos
            soft_state.verts_vel_prev[k, i_v, i_b] = qd.Vector.zero(gs.qd_float, 3)
        soft_state.verts_pos_stage_start[i_v, i_b] = pos
        soft_state.verts_vel_stage_start[i_v, i_b] = qd.Vector.zero(gs.qd_float, 3)
        soft_state.verts_is_fixed[i_v, i_b] = False
        soft_state.verts_target[i_v, i_b] = pos
        soft_state.verts_contact_force[i_v, i_b] = qd.Vector.zero(gs.qd_float, 3)


@qd.func
def func_soft_dof(i_v, k, soft_info: MochiSoftInfo):
    return soft_info.dof_start[None] + 3 * i_v + k


@qd.func
def func_read_soft_vec(src: qd.Tensor, i_v, i_b, soft_info: MochiSoftInfo):
    i_d = func_soft_dof(i_v, 0, soft_info)
    return qd.Vector([src[i_d, i_b], src[i_d + 1, i_b], src[i_d + 2, i_b]], dt=gs.qd_float)


@qd.func
def func_add_soft_vec(dst: qd.Tensor, i_v, i_b, value, soft_info: MochiSoftInfo):
    i_d = func_soft_dof(i_v, 0, soft_info)
    for k in qd.static(range(3)):
        qd.atomic_add(dst[i_d + k, i_b], value[k])


@qd.func
def func_deformation_gradient(x0, x1, x2, x3, Dm_inv):
    return qd.Matrix.cols([x0 - x3, x1 - x3, x2 - x3]) @ Dm_inv


@qd.func
def func_shape_gradients(Dm_inv):
    """Rest gradients of the four linear shape functions (rows of Dm^-1 and minus their sum)."""
    g0 = qd.Vector([Dm_inv[0, 0], Dm_inv[0, 1], Dm_inv[0, 2]], dt=gs.qd_float)
    g1 = qd.Vector([Dm_inv[1, 0], Dm_inv[1, 1], Dm_inv[1, 2]], dt=gs.qd_float)
    g2 = qd.Vector([Dm_inv[2, 0], Dm_inv[2, 1], Dm_inv[2, 2]], dt=gs.qd_float)
    return g0, g1, g2, -(g0 + g1 + g2)


# ------------------------------------------------------------------------------------
# ----------------------------------- time integration -------------------------------
# ------------------------------------------------------------------------------------


@qd.func
def func_soft_step_start(
    i_b_env,
    per_env: qd.template(),
    envs: qd.types.ndarray(),
    n_envs: qd.types.ndarray(),
    mochi_state: MochiState,
    soft_info: MochiSoftInfo,
    soft_state: MochiSoftState,
    rigid_config: qd.template(),
    mochi_config: qd.template(),
):
    """Shift the vertex history and build the step-start (= stage-start) positions and velocities, which also warm
    start the solve; snapshot the stage-start deformation gradients for the stiffness damping. Runs after the rigid
    step start (which advances the history counter)."""
    n_verts = soft_state.verts_pos.shape[0]
    n_elems = soft_state.elems_F_stage_start.shape[0]
    _B = soft_state.verts_pos.shape[1]

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_v, i_slot in qd.ndrange(n_verts, n_envs[None]) if qd.static(not per_env) else qd.ndrange(n_verts, 1):
        i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
        soft_state.verts_pos_prev[1, i_v, i_b] = soft_state.verts_pos_prev[0, i_v, i_b]
        soft_state.verts_vel_prev[1, i_v, i_b] = soft_state.verts_vel_prev[0, i_v, i_b]
        soft_state.verts_pos_prev[0, i_v, i_b] = soft_state.verts_pos[i_v, i_b]
        soft_state.verts_vel_prev[0, i_v, i_b] = soft_state.verts_vel[i_v, i_b]
        x1 = soft_state.verts_pos[i_v, i_b]
        v1 = soft_state.verts_vel[i_v, i_b]
        pos = x1
        vel = v1
        if qd.static(mochi_config.integrator == INTEGRATOR.BDF2):  # noqa: SIM102
            if mochi_state.n_hist[i_b] >= 2:
                pos = x1 + BDF2_ALPHA_2 * (soft_state.verts_pos_prev[1, i_v, i_b] - x1)
                vel = v1 + BDF2_ALPHA_2 * (soft_state.verts_vel_prev[1, i_v, i_b] - v1)
        soft_state.verts_pos_stage_start[i_v, i_b] = pos
        soft_state.verts_vel_stage_start[i_v, i_b] = vel
        # A fixed vertex takes its prescribed end-of-step position at once (its rows of the Newton system are
        # identities); its velocity then follows by finite differences like that of any other vertex.
        if soft_state.verts_is_fixed[i_v, i_b]:
            pos = soft_state.verts_target[i_v, i_b]
        soft_state.verts_pos[i_v, i_b] = pos

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_el, i_slot in qd.ndrange(n_elems, n_envs[None]) if qd.static(not per_env) else qd.ndrange(n_elems, 1):
        i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
        v = soft_info.elems_v[i_el]
        soft_state.elems_F_stage_start[i_el, i_b] = func_deformation_gradient(
            soft_state.verts_pos_stage_start[v[0], i_b],
            soft_state.verts_pos_stage_start[v[1], i_b],
            soft_state.verts_pos_stage_start[v[2], i_b],
            soft_state.verts_pos_stage_start[v[3], i_b],
            soft_info.elems_Dm_inv[i_el],
        )


@qd.kernel
def kernel_soft_step_start(
    mochi_state: MochiState,
    soft_info: MochiSoftInfo,
    soft_state: MochiSoftState,
    rigid_config: qd.template(),
    mochi_config: qd.template(),
):
    func_soft_step_start(
        0,
        False,
        mochi_state.all_envs,
        mochi_state.n_envs_all,
        mochi_state,
        soft_info,
        soft_state,
        rigid_config,
        mochi_config,
    )


@qd.func
def func_soft_post_stage(
    i_b_env,
    per_env: qd.template(),
    envs: qd.types.ndarray(),
    n_envs: qd.types.ndarray(),
    mochi_state: MochiState,
    soft_info: MochiSoftInfo,
    soft_state: MochiSoftState,
    rigid_config: qd.template(),
):
    """Finite-difference vertex velocities over the stage; a diverged environment falls back to the rest shape at
    rest. Neither the stage start nor the previous step is a guaranteed-safe deformable state, whereas the rest shape
    is free of inverted or degenerate elements by construction."""
    n_verts = soft_state.verts_pos.shape[0]
    _B = soft_state.verts_pos.shape[1]
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_v, i_slot in qd.ndrange(n_verts, n_envs[None]) if qd.static(not per_env) else qd.ndrange(n_verts, 1):
        i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
        if mochi_state.status[i_b] == SOLVE_STATUS.DIVERGED:
            soft_state.verts_pos[i_v, i_b] = soft_info.verts_rest[i_v]
            soft_state.verts_vel[i_v, i_b] = qd.Vector.zero(gs.qd_float, 3)
            continue
        h = mochi_state.dt_stage[i_b]
        soft_state.verts_vel[i_v, i_b] = (
            soft_state.verts_pos[i_v, i_b] - soft_state.verts_pos_stage_start[i_v, i_b]
        ) / h


@qd.kernel
def kernel_soft_post_stage(
    mochi_state: MochiState,
    soft_info: MochiSoftInfo,
    soft_state: MochiSoftState,
    rigid_config: qd.template(),
):
    func_soft_post_stage(
        0,
        False,
        mochi_state.all_envs,
        mochi_state.n_envs_all,
        mochi_state,
        soft_info,
        soft_state,
        rigid_config,
    )


# ------------------------------------------------------------------------------------
# --------------------------------------- Newton -------------------------------------
# ------------------------------------------------------------------------------------


@qd.func
def func_soft_update_conv_weights(
    i_b_env,
    per_env: qd.template(),
    envs: qd.types.ndarray(),
    n_envs: qd.types.ndarray(),
    mochi_info: MochiInfo,
    mochi_state: MochiState,
    soft_info: MochiSoftInfo,
    soft_state: MochiSoftState,
    rigid_config: qd.template(),
):
    """Convergence weights of the vertex degrees of freedom: 1 / (a_ref^2 M_entity m_vertex) with the lumped vertex
    mass, so that a unit weighted residual norm means a unit acceleration error."""
    n_verts = soft_state.verts_pos.shape[0]
    _B = soft_state.verts_pos.shape[1]
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_v, i_slot in qd.ndrange(n_verts, n_envs[None]) if qd.static(not per_env) else qd.ndrange(n_verts, 1):
        i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
        # A massless vertex has no acceleration scale to normalize by; see kernel_update_conv_weights.
        mass = soft_info.entities_mass[soft_info.verts_entity_idx[i_v]] * soft_info.verts_mass[i_v]
        w = gs.qd_float(1.0)
        if mass > 0.0:
            a_ref = qd.max(1.0, mochi_info.gravity[i_b].norm())
            w = 1.0 / (a_ref * a_ref * mass)
        for k in qd.static(range(3)):
            mochi_state.conv_w[func_soft_dof(i_v, k, soft_info), i_b] = w


@qd.kernel
def kernel_soft_update_conv_weights(
    mochi_info: MochiInfo,
    mochi_state: MochiState,
    soft_info: MochiSoftInfo,
    soft_state: MochiSoftState,
    rigid_config: qd.template(),
):
    func_soft_update_conv_weights(
        0,
        False,
        mochi_state.all_envs,
        mochi_state.n_envs_all,
        mochi_info,
        mochi_state,
        soft_info,
        soft_state,
        rigid_config,
    )


@qd.func
def func_soft_store_ls_ref(
    i_b_env,
    per_env: qd.template(),
    envs: qd.types.ndarray(),
    n_envs: qd.types.ndarray(),
    mochi_state: MochiState,
    soft_state: MochiSoftState,
    rigid_config: qd.template(),
    only_done,
):
    """Take the current vertex positions as the line search reference of the active environments (of those that just
    accepted an iterate when only_done)."""
    n_verts = soft_state.verts_pos.shape[0]
    _B = soft_state.verts_pos.shape[1]
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_v, i_slot in qd.ndrange(n_verts, n_envs[None]) if qd.static(not per_env) else qd.ndrange(n_verts, 1):
        i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
        is_ref = mochi_state.is_active[i_b]
        if only_done:
            is_ref = is_ref and mochi_state.ls_is_done[i_b]
        if is_ref:
            soft_state.verts_pos_ls_ref[i_v, i_b] = soft_state.verts_pos[i_v, i_b]


@qd.kernel
def kernel_soft_store_ls_ref(
    mochi_state: MochiState,
    soft_state: MochiSoftState,
    rigid_config: qd.template(),
    only_done: qd.template(),
):
    func_soft_store_ls_ref(
        0,
        False,
        mochi_state.all_envs,
        mochi_state.n_envs_all,
        mochi_state,
        soft_state,
        rigid_config,
        only_done,
    )


@qd.func
def func_soft_apply_increment(
    i_b_env,
    per_env: qd.template(),
    envs: qd.types.ndarray(),
    n_envs: qd.types.ndarray(),
    mochi_state: MochiState,
    soft_info: MochiSoftInfo,
    soft_state: MochiSoftState,
    rigid_config: qd.template(),
):
    """Trial vertex positions of the environments still searching: reference minus the scaled Newton step."""
    n_verts = soft_state.verts_pos.shape[0]
    _B = soft_state.verts_pos.shape[1]
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_v, i_slot in qd.ndrange(n_verts, n_envs[None]) if qd.static(not per_env) else qd.ndrange(n_verts, 1):
        i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
        if not func_is_env_active(i_b, mochi_state, True):
            continue
        if soft_state.verts_is_fixed[i_v, i_b]:
            continue
        dx = func_read_soft_vec(mochi_state.dx, i_v, i_b, soft_info)
        soft_state.verts_pos[i_v, i_b] = soft_state.verts_pos_ls_ref[i_v, i_b] - mochi_state.ls_alpha[i_b] * dx


@qd.kernel
def kernel_soft_apply_increment(
    mochi_state: MochiState,
    soft_info: MochiSoftInfo,
    soft_state: MochiSoftState,
    rigid_config: qd.template(),
):
    func_soft_apply_increment(
        0,
        False,
        mochi_state.all_envs,
        mochi_state.n_envs_all,
        mochi_state,
        soft_info,
        soft_state,
        rigid_config,
    )


# ------------------------------------------------------------------------------------
# -------------------------------------- assembly ------------------------------------
# ------------------------------------------------------------------------------------


@qd.func
def func_soft_zero_assembly(
    i_b_env,
    per_env: qd.template(),
    envs: qd.types.ndarray(),
    n_envs: qd.types.ndarray(),
    mochi_state: MochiState,
    soft_state: MochiSoftState,
    rigid_config: qd.template(),
    assem_dres,
    skip_ls_done,
    record: qd.template(),
):
    n_verts = soft_state.verts_pos.shape[0]
    max_pairs = soft_state.pair_entity_a.shape[0]
    _B = soft_state.verts_pos.shape[1]

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.ALL))
    for i_slot in range(n_envs[None]) if qd.static(not per_env) else range(1):
        i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
        if func_is_env_active(i_b, mochi_state, skip_ls_done):
            soft_state.n_soft_hits[i_b] = 0
            soft_state.n_sc_hits[i_b] = 0
            soft_state.n_pc_hits[i_b] = 0
    if qd.static((not isinstance(assem_dres, (bool, int)) or bool(assem_dres)) or record):
        qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
        for i_v, i_slot in qd.ndrange(n_verts, n_envs[None]) if qd.static(not per_env) else qd.ndrange(n_verts, 1):
            i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
            if func_is_env_active(i_b, mochi_state, skip_ls_done):
                if assem_dres:
                    soft_state.verts_H_diag[i_v, i_b] = qd.Matrix.zero(gs.qd_float, 3, 3)
                if qd.static(record):
                    soft_state.verts_contact_force[i_v, i_b] = qd.Vector.zero(gs.qd_float, 3)
    # The Hessian is zeroed on full assemblies only. When the caller binds the flag to a compile-time constant (the
    # pipeline's zeroing kernel), the residual-only variant compiles these loops out entirely; the monolith and graph
    # step kernels bind a runtime flag and keep the loops, testing the flag per element (a loop nested under a runtime
    # condition is not offloaded as a parallel task and would run serially on one thread).
    if qd.static(not isinstance(assem_dres, (bool, int)) or bool(assem_dres)):
        n_csr = soft_state.csr_values.shape[0]
        qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
        for j, i_slot in qd.ndrange(n_csr, n_envs[None]) if qd.static(not per_env) else qd.ndrange(n_csr, 1):
            i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
            if assem_dres and func_is_env_active(i_b, mochi_state, skip_ls_done):
                soft_state.csr_values[j, i_b] = 0.0
        # The rod blocks kept for the banded preconditioner are overwritten by the assembly kernels (padding elements
        # keep their zero allocation); only the accumulated twist diagonal needs zeroing.
        n_rod_elems = soft_state.rod_elems_H.shape[0]
        qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
        for i_r, i_slot in (
            qd.ndrange(n_rod_elems, n_envs[None]) if qd.static(not per_env) else qd.ndrange(n_rod_elems, 1)
        ):
            i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
            if assem_dres and func_is_env_active(i_b, mochi_state, skip_ls_done):
                soft_state.rod_elems_twist_pcg[i_r, i_b] = 0.0
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_p, i_slot in qd.ndrange(max_pairs, n_envs[None]) if qd.static(not per_env) else qd.ndrange(max_pairs, 1):
        i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
        if func_is_env_active(i_b, mochi_state, skip_ls_done) and i_p < soft_state.n_pairs[i_b]:
            soft_state.acc_f[i_p, i_b] = qd.Vector.zero(gs.qd_float, 3)
            soft_state.acc_q[i_p, i_b] = qd.Vector.zero(gs.qd_float, 3)
            soft_state.acc_D[i_p, i_b] = qd.Vector.zero(gs.qd_float, 6)
            soft_state.acc_SD[i_p, i_b] = qd.Matrix.zero(gs.qd_float, 3, 3)
            soft_state.acc_SDS[i_p, i_b] = qd.Vector.zero(gs.qd_float, 6)
            soft_state.acc_obj[i_p, i_b] = 0.0
            soft_state.n_hits[i_p, i_b] = 0


@qd.kernel
def kernel_soft_zero_assembly(
    mochi_state: MochiState,
    soft_state: MochiSoftState,
    rigid_config: qd.template(),
    assem_dres: qd.template(),
    skip_ls_done: qd.i32,
    record: qd.template(),
):
    func_soft_zero_assembly(
        0,
        False,
        mochi_state.all_envs,
        mochi_state.n_envs_all,
        mochi_state,
        soft_state,
        rigid_config,
        assem_dres,
        skip_ls_done,
        record,
    )


@qd.func
def func_soft_assemble_elements(
    i_b_env,
    per_env: qd.template(),
    envs: qd.types.ndarray(),
    n_envs: qd.types.ndarray(),
    mochi_info: MochiInfo,
    mochi_state: MochiState,
    soft_info: MochiSoftInfo,
    soft_state: MochiSoftState,
    rigid_config: qd.template(),
    assem_obj: qd.template(),
    assem_res: qd.template(),
    assem_dres,
    skip_ls_done,
):
    """Inertia, gravity, mass damping, elastic stress and stiffness damping of every tetrahedron of the running
    environments: residual into the vertex degrees of freedom, 12x12 tangent block per element (positive-semidefinite
    by construction) and its diagonal 3x3 blocks into the vertex preconditioner."""
    n_elems = soft_info.elems_v.shape[0]
    _B = soft_state.verts_pos.shape[1]
    EPS = mochi_info.EPS[None]

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_el, i_slot in qd.ndrange(n_elems, n_envs[None]) if qd.static(not per_env) else qd.ndrange(n_elems, 1):
        i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
        if not func_is_env_active(i_b, mochi_state, skip_ls_done):
            continue
        vol = soft_info.elems_vol[i_el]
        if vol <= 0.0:
            # Padding element of a scene without solids.
            continue
        i_e = soft_info.elems_entity_idx[i_el]
        v = soft_info.elems_v[i_el]
        h = mochi_state.dt_stage[i_b]
        rho = soft_info.entities_rho[i_e]
        mu = soft_info.entities_mu[i_e]
        lam = soft_info.entities_lam[i_e]
        model = soft_info.entities_model[i_e]
        Dm_inv = soft_info.elems_Dm_inv[i_el]

        x = qd.Matrix.zero(gs.qd_float, 4, 3)
        a = qd.Matrix.zero(gs.qd_float, 4, 3)  # x - (x_ss + h v_ss)
        b = qd.Matrix.zero(gs.qd_float, 4, 3)  # x - x_ss
        for f in qd.static(range(4)):
            pos = soft_state.verts_pos[v[f], i_b]
            pos_ss = soft_state.verts_pos_stage_start[v[f], i_b]
            vel_ss = soft_state.verts_vel_stage_start[v[f], i_b]
            for k in qd.static(range(3)):
                x[f, k] = pos[k]
                a[f, k] = pos[k] - pos_ss[k] - h * vel_ss[k]
                b[f, k] = pos[k] - pos_ss[k]

        # Inertia and mass damping through the consistent mass matrix (rho V / 20) [2 1 1 1; ...] (x) I.
        m_unit = rho * vol / 20.0
        c_inertia = m_unit / (h * h)
        c_damping = m_unit * soft_info.entities_mass_damping[i_e] / h
        energy = gs.qd_float(0.0)
        res = qd.Matrix.zero(gs.qd_float, 4, 3)
        for f in qd.static(range(4)):
            for g in qd.static(range(4)):
                m_fg = gs.qd_float(CONSISTENT_MASS[f][g])
                for k in qd.static(range(3)):
                    res[f, k] += m_fg * (c_inertia * a[g, k] + c_damping * b[g, k])
                    if qd.static(assem_obj):
                        energy += 0.5 * m_fg * (c_inertia * a[f, k] * a[g, k] + c_damping * b[f, k] * b[g, k])

        # Gravity: one quadrature point, each node carries a quarter of the weight.
        if soft_info.entities_has_gravity[i_e]:
            gravity = mochi_info.gravity[i_b]
            for f in qd.static(range(4)):
                for k in qd.static(range(3)):
                    res[f, k] -= 0.25 * rho * vol * gravity[k]
                    if qd.static(assem_obj):
                        energy -= 0.25 * rho * vol * gravity[k] * x[f, k]

        # Elastic stress: F = Ds Dm^-1, residual V P grad N_f.
        F = func_deformation_gradient(
            soft_state.verts_pos[v[0], i_b],
            soft_state.verts_pos[v[1], i_b],
            soft_state.verts_pos[v[2], i_b],
            soft_state.verts_pos[v[3], i_b],
            Dm_inv,
        )
        g0, g1, g2, g3 = func_shape_gradients(Dm_inv)
        grads = qd.Matrix.rows([g0, g1, g2, g3])
        P = func_elastic_stress(model, F, mu, lam)
        if qd.static(assem_obj):
            energy += vol * func_elastic_energy(model, F, mu, lam)
        kappa = soft_info.entities_stiffness_damping[i_e] / h
        has_stiffness_damping = kappa > 0.0
        if has_stiffness_damping:
            energy_damping, S_visc = func_stiffness_damping_stress(
                F, soft_state.elems_F_stage_start[i_el, i_b], mu, lam, kappa
            )
            P += F @ S_visc
            if qd.static(assem_obj):
                energy += vol * energy_damping
        for f in qd.static(range(4)):
            g_f = qd.Vector([grads[f, 0], grads[f, 1], grads[f, 2]], dt=gs.qd_float)
            Pg = vol * (P @ g_f)
            for k in qd.static(range(3)):
                res[f, k] += Pg[k]

        if qd.static(assem_res):
            for f in qd.static(range(4)):
                i_d = func_soft_dof(v[f], 0, soft_info)
                for k in qd.static(range(3)):
                    qd.atomic_add(mochi_state.res[i_d + k, i_b], res[f, k])
        if qd.static(assem_obj):
            qd.atomic_add(mochi_state.obj[i_b], energy)

        if assem_dres:
            # Elastic stiffness per node block (no 9x9 tangent), then the damping and the consistent mass.
            K = func_tet_stiffness(model, F, mu, lam, EPS, True, grads, vol)
            for f in qd.static(range(4)):
                g_f = qd.Vector([grads[f, 0], grads[f, 1], grads[f, 2]], dt=gs.qd_float)
                for g in qd.static(range(4)):
                    g_g = qd.Vector([grads[g, 0], grads[g, 1], grads[g, 2]], dt=gs.qd_float)
                    if has_stiffness_damping:
                        damping = vol * func_stiffness_damping_block(F, g_f, g_g, mu, lam, kappa)
                        for r in qd.static(range(3)):
                            for c in qd.static(range(3)):
                                K[3 * f + r, 3 * g + c] += damping[r, c]
                    m_fg = gs.qd_float(CONSISTENT_MASS[f][g]) * (c_inertia + c_damping)
                    for k in qd.static(range(3)):
                        K[3 * f + k, 3 * g + k] += m_fg
                    if qd.static(f == g):
                        block = qd.Matrix.zero(gs.qd_float, 3, 3)
                        for r in qd.static(range(3)):
                            for c in qd.static(range(3)):
                                block[r, c] = K[3 * f + r, 3 * f + c]
                        qd.atomic_add(soft_state.verts_H_diag[v[f], i_b], block)
            # scatter per vertex block: row start + block position + column
            for f in qd.static(range(4)):
                for r in qd.static(range(3)):
                    row_start = soft_info.csr_start[3 * v[f] + r]
                    for g in qd.static(range(4)):
                        pos = row_start + soft_info.elems_csr_block[i_el, 4 * f + g]
                        for c in qd.static(range(3)):
                            qd.atomic_add(soft_state.csr_values[pos + c, i_b], K[3 * f + r, 3 * g + c])


@qd.kernel
def kernel_soft_assemble_elements(
    mochi_info: MochiInfo,
    mochi_state: MochiState,
    soft_info: MochiSoftInfo,
    soft_state: MochiSoftState,
    rigid_config: qd.template(),
    assem_obj: qd.template(),
    assem_res: qd.template(),
    assem_dres: qd.i32,
    skip_ls_done: qd.i32,
):
    func_soft_assemble_elements(
        0,
        False,
        mochi_state.all_envs,
        mochi_state.n_envs_all,
        mochi_info,
        mochi_state,
        soft_info,
        soft_state,
        rigid_config,
        assem_obj,
        assem_res,
        assem_dres,
        skip_ls_done,
    )


@qd.func
def func_soft_dirichlet(
    i_b_env,
    per_env: qd.template(),
    envs: qd.types.ndarray(),
    n_envs: qd.types.ndarray(),
    mochi_state: MochiState,
    soft_info: MochiSoftInfo,
    soft_state: MochiSoftState,
    rigid_config: qd.template(),
    skip_ls_done,
):
    """Zero the residual of the fixed vertices (their rows of the Newton system are identities)."""
    n_verts = soft_state.verts_pos.shape[0]
    _B = soft_state.verts_pos.shape[1]
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_v, i_slot in qd.ndrange(n_verts, n_envs[None]) if qd.static(not per_env) else qd.ndrange(n_verts, 1):
        i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
        if func_is_env_active(i_b, mochi_state, skip_ls_done) and soft_state.verts_is_fixed[i_v, i_b]:
            for k in qd.static(range(3)):
                mochi_state.res[func_soft_dof(i_v, k, soft_info), i_b] = 0.0


@qd.kernel
def kernel_soft_dirichlet(
    mochi_state: MochiState,
    soft_info: MochiSoftInfo,
    soft_state: MochiSoftState,
    rigid_config: qd.template(),
    skip_ls_done: qd.i32,
):
    func_soft_dirichlet(
        0,
        False,
        mochi_state.all_envs,
        mochi_state.n_envs_all,
        mochi_state,
        soft_info,
        soft_state,
        rigid_config,
        skip_ls_done,
    )


# ------------------------------------------------------------------------------------
# --------------------------------------- contact ------------------------------------
# ------------------------------------------------------------------------------------


@qd.func
def func_soft_conservative_bounds(
    i_b_env,
    per_env: qd.template(),
    envs: qd.types.ndarray(),
    n_envs: qd.types.ndarray(),
    mochi_info: MochiInfo,
    soft_info: MochiSoftInfo,
    soft_state: MochiSoftState,
    rigid_config: qd.template(),
):
    """World bounds of every deformable entity that hold for the whole step, from the stage-start positions and
    velocities of its vertices."""
    n_verts = soft_state.verts_pos.shape[0]
    n_entities = soft_state.entities_step_aabb_min.shape[0]
    _B = soft_state.verts_pos.shape[1]
    dt = mochi_info.dt[None]
    margin = mochi_info.broadphase_margin[None]

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_e, i_slot in qd.ndrange(n_entities, n_envs[None]) if qd.static(not per_env) else qd.ndrange(n_entities, 1):
        i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
        soft_state.entities_step_aabb_min[i_e, i_b] = qd.Vector([gs.qd_float(1e30)] * 3, dt=gs.qd_float)
        soft_state.entities_step_aabb_max[i_e, i_b] = qd.Vector([gs.qd_float(-1e30)] * 3, dt=gs.qd_float)
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_v, i_slot in qd.ndrange(n_verts, n_envs[None]) if qd.static(not per_env) else qd.ndrange(n_verts, 1):
        i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
        i_e = soft_info.verts_entity_idx[i_v]
        speed = soft_state.verts_vel_stage_start[i_v, i_b].norm()
        speed = CONSERVATIVE_SPEED_SCALE * speed + CONSERVATIVE_MAX_ACCEL * dt
        if soft_info.entities_has_gravity[i_e]:
            speed += mochi_info.gravity[i_b].norm() * dt
        pad = margin + speed * dt
        pos = soft_state.verts_pos_stage_start[i_v, i_b]
        for k in qd.static(range(3)):
            qd.atomic_min(soft_state.entities_step_aabb_min[i_e, i_b][k], pos[k] - pad)
            qd.atomic_max(soft_state.entities_step_aabb_max[i_e, i_b][k], pos[k] + pad)


@qd.kernel
def kernel_soft_conservative_bounds(
    mochi_state: MochiState,
    mochi_info: MochiInfo,
    soft_info: MochiSoftInfo,
    soft_state: MochiSoftState,
    rigid_config: qd.template(),
):
    func_soft_conservative_bounds(
        0,
        False,
        mochi_state.all_envs,
        mochi_state.n_envs_all,
        mochi_info,
        soft_info,
        soft_state,
        rigid_config,
    )


@qd.func
def func_soft_broadphase(
    i_b_env,
    per_env: qd.template(),
    envs: qd.types.ndarray(),
    n_envs: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    mochi_info: MochiInfo,
    mochi_state: MochiState,
    contact_state: MochiContactState,
    soft_info: MochiSoftInfo,
    soft_state: MochiSoftState,
    rigid_config: qd.template(),
    errno: qd.Tensor,
):
    """Enumerate the (deformable entity, collider geom) pairs whose conservative bounds overlap within the step."""
    n_entities = soft_state.entities_step_aabb_min.shape[0]
    n_geoms = dyn_state.geoms.pos.shape[0]
    _B = soft_state.verts_pos.shape[1]
    max_pairs = soft_state.pair_entity_a.shape[0]

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.ALL))
    for i_slot in range(n_envs[None]) if qd.static(not per_env) else range(1):
        i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
        soft_state.n_pairs[i_b] = 0

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_e, i_gb, i_slot in (
        qd.ndrange(n_entities, n_geoms, n_envs[None]) if qd.static(not per_env) else qd.ndrange(n_entities, n_geoms, 1)
    ):
        i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
        if not mochi_state.is_active[i_b]:
            continue
        if soft_info.entities_sample_end[i_e] <= soft_info.entities_sample_start[i_e]:
            continue
        if mochi_info.geoms.collider_type[i_gb] == COLLIDER_TYPE.NONE:
            continue
        i_lb = dyn_info.geoms.link_idx[i_gb]
        if not soft_info.entities_links_pair_enabled[i_e, i_lb]:
            continue
        if mochi_info.geoms.collider_type[i_gb] != COLLIDER_TYPE.PLANE:
            band = contact_state.links_step_pad[i_lb, i_b] + mochi_info.geoms.penalty_threshold[i_gb]
            geom_min = dyn_state.geoms.aabb_min[i_gb, i_b] - band
            geom_max = dyn_state.geoms.aabb_max[i_gb, i_b] + band
            if (soft_state.entities_step_aabb_max[i_e, i_b] < geom_min).any():
                continue
            if (soft_state.entities_step_aabb_min[i_e, i_b] > geom_max).any():
                continue
        i_p = qd.atomic_add(soft_state.n_pairs[i_b], 1)
        if i_p < max_pairs:
            soft_state.pair_entity_a[i_p, i_b] = i_e
            soft_state.pair_link_b[i_p, i_b] = i_lb
            soft_state.pair_geom_b[i_p, i_b] = i_gb
        else:
            qd.atomic_or(errno[i_b], array_class.ErrorCode.OVERFLOW_MOCHI_CONTACT_PAIRS)

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.ALL))
    for i_slot in range(n_envs[None]) if qd.static(not per_env) else range(1):
        i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
        soft_state.n_pairs[i_b] = qd.min(soft_state.n_pairs[i_b], max_pairs)


@qd.kernel
def kernel_soft_broadphase(
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    mochi_info: MochiInfo,
    mochi_state: MochiState,
    contact_state: MochiContactState,
    soft_info: MochiSoftInfo,
    soft_state: MochiSoftState,
    rigid_config: qd.template(),
    errno: qd.Tensor,
):
    func_soft_broadphase(
        0,
        False,
        mochi_state.all_envs,
        mochi_state.n_envs_all,
        dyn_state,
        dyn_info,
        mochi_info,
        mochi_state,
        contact_state,
        soft_info,
        soft_state,
        rigid_config,
        errno,
    )


@qd.func
def func_soft_contact_eval(
    i_b_env,
    per_env: qd.template(),
    envs: qd.types.ndarray(),
    n_envs: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    sdf_info: array_class.SDFInfo,
    mochi_info: MochiInfo,
    mochi_state: MochiState,
    soft_info: MochiSoftInfo,
    soft_state: MochiSoftState,
    hit_readback: MochiHitReadback,
    rigid_config: qd.template(),
    mochi_config: qd.template(),
    max_samples_per_entity: int,
    assem_res: qd.template(),
    assem_dres,
    skip_ls_done,
    record: qd.template(),
    errno: qd.Tensor,
):
    """Evaluate every boundary sample of every candidate pair against its collider at the current vertex positions:
    residual on the three vertices (barycentric weights), per-sample matrix D = -w df/dp recorded for the vertex and
    coupling blocks, and the rigid-side accumulators of the pair."""
    max_pairs = soft_state.pair_entity_a.shape[0]
    max_hits = soft_state.hit_sample.shape[0]
    _B = soft_state.verts_pos.shape[1]
    EPS = mochi_info.EPS[None]

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_p, i_s_, i_slot in (
        qd.ndrange(max_pairs, max_samples_per_entity, n_envs[None])
        if qd.static(not per_env)
        else qd.ndrange(max_pairs, max_samples_per_entity, 1)
    ):
        i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
        if not func_is_env_active(i_b, mochi_state, skip_ls_done):
            continue
        if i_p >= soft_state.n_pairs[i_b]:
            continue
        i_e = soft_state.pair_entity_a[i_p, i_b]
        i_s = soft_info.entities_sample_start[i_e] + i_s_
        if i_s >= soft_info.entities_sample_end[i_e]:
            continue
        i_lb = soft_state.pair_link_b[i_p, i_b]
        i_gb = soft_state.pair_geom_b[i_p, i_b]

        tri = soft_info.samples_tri[i_s]
        bary = soft_info.samples_bary[i_s]
        x0 = soft_state.verts_pos[tri[0], i_b]
        x1 = soft_state.verts_pos[tri[1], i_b]
        x2 = soft_state.verts_pos[tri[2], i_b]
        pos = bary[0] * x0 + bary[1] * x1 + bary[2] * x2
        thr = mochi_info.geoms.penalty_threshold[i_gb]
        h = mochi_info.geoms.penalty_smoothing_half_distance[i_gb]
        # Contact range: the penalty and its derivatives vanish beyond the threshold (mochi's detection range).
        band = thr
        if mochi_info.geoms.collider_type[i_gb] != COLLIDER_TYPE.PLANE:
            if (pos < dyn_state.geoms.aabb_min[i_gb, i_b] - band).any():
                continue
            if (pos > dyn_state.geoms.aabb_max[i_gb, i_b] + band).any():
                continue

        pos_g = dyn_state.geoms.pos[i_gb, i_b]
        quat_g = dyn_state.geoms.quat[i_gb, i_b]
        pos_geom = gu.qd_inv_transform_by_trans_quat(pos, pos_g, quat_g)
        is_valid, d, grad = query_collider(i_gb, pos_geom, dyn_info.geoms, mochi_info.geoms, sdf_info, mochi_config)
        if not is_valid or d > band:
            continue

        # Colliding normal of the deformed triangle and stage displacement of the sample, in the collider frame. Shell
        # samples carry no orientation: the collider gradient serves as their normal.
        normal_world = gu.qd_normalize((x1 - x0).cross(x2 - x0), EPS)
        normal_geom = gu.qd_inv_transform_by_quat(normal_world, quat_g)
        if soft_info.entities_kind[i_e] != SOFT_KIND_SOLID:
            normal_geom = -gu.qd_normalize(grad, EPS)
        pos_start = (
            bary[0] * soft_state.verts_pos_stage_start[tri[0], i_b]
            + bary[1] * soft_state.verts_pos_stage_start[tri[1], i_b]
            + bary[2] * soft_state.verts_pos_stage_start[tri[2], i_b]
        )
        pos_geom_start = gu.qd_inv_transform_by_trans_quat(
            pos_start, mochi_state.geoms_pos_stage_start[i_gb, i_b], mochi_state.geoms_quat_stage_start[i_gb, i_b]
        )
        p_rel = pos_geom - pos_geom_start
        d_start = d - grad.dot(p_rel)

        is_static_b = not mochi_info.links.is_dynamic[i_lb]
        k = qd.sqrt(soft_info.entities_penalty_coefficient[i_e] * mochi_info.geoms.penalty_coefficient[i_gb])
        falloff = qd.sqrt(soft_info.entities_friction_falloff_vel[i_e] * mochi_info.geoms.friction_falloff_vel[i_gb])
        if is_static_b:
            k = soft_info.entities_penalty_coefficient[i_e]
            falloff = soft_info.entities_friction_falloff_vel[i_e]
        mu = qd.sqrt(soft_info.entities_friction[i_e] * mochi_info.geoms.friction[i_gb])
        c_visc = qd.sqrt(soft_info.entities_viscous_friction[i_e] * mochi_info.geoms.viscous_friction[i_gb])
        c_ndamp = qd.sqrt(
            soft_info.entities_normal_viscous_damping[i_e] * mochi_info.geoms.normal_viscous_damping[i_gb]
        )
        max_align = mochi_info.geoms.max_alignment_normals[i_gb]

        energy, force_geom, dforce_geom, _ = collision_response(
            d,
            grad,
            normal_geom,
            p_rel,
            d_start,
            k,
            h,
            thr,
            mu,
            falloff,
            c_visc,
            c_ndamp,
            max_align,
            mochi_state.dt_stage[i_b],
            EPS,
            mochi_config,
        )

        w = soft_info.samples_weight[i_s]
        R_g = gu.qd_quat_to_R(quat_g, EPS)
        force = R_g @ force_geom
        D = -w * (R_g @ dforce_geom @ R_g.transpose())
        r_b = pos - dyn_state.links.pos[i_lb, i_b]
        S_b = skew(r_b)

        if qd.static(assem_res):
            for i in qd.static(range(3)):
                func_add_soft_vec(mochi_state.res, tri[i], i_b, -(w * bary[i]) * force, soft_info)
        qd.atomic_add(soft_state.acc_f[i_p, i_b], w * force)
        qd.atomic_add(soft_state.acc_q[i_p, i_b], w * r_b.cross(force))
        qd.atomic_add(soft_state.acc_obj[i_p, i_b], w * energy)
        qd.atomic_add(soft_state.n_hits[i_p, i_b], 1)

        # The three per-pair Hessian sums are read by kernel_soft_pairs_to_blocks under the same flag, and they carry
        # most of the atomic traffic of this kernel: the line search re-evaluates contact for the residual alone.
        if assem_dres:
            qd.atomic_add(soft_state.acc_D[i_p, i_b], func_mat3_to_sym6(D))
            qd.atomic_add(soft_state.acc_SD[i_p, i_b], S_b @ D)
            qd.atomic_add(soft_state.acc_SDS[i_p, i_b], func_mat3_to_sym6(S_b @ D @ S_b))
            for i in qd.static(range(3)):
                qd.atomic_add(soft_state.verts_H_diag[tri[i], i_b], (bary[i] * bary[i]) * D)
        if assem_dres or record:
            i_h = qd.atomic_add(soft_state.n_soft_hits[i_b], 1)
            if i_h < max_hits:
                soft_state.hit_sample[i_h, i_b] = i_s
                soft_state.hit_link_b[i_h, i_b] = -1 if is_static_b else i_lb
                soft_state.hit_r_b[i_h, i_b] = r_b
                soft_state.hit_D[i_h, i_b] = func_mat3_to_sym6(D)
                if qd.static(record):
                    hit_readback.soft_hit_geom_b[i_h, i_b] = soft_state.pair_geom_b[i_p, i_b]
                    hit_readback.soft_hit_force[i_h, i_b] = w * force
                    hit_readback.soft_hit_pos[i_h, i_b] = pos
                    hit_readback.soft_hit_normal[i_h, i_b] = gu.qd_normalize(R_g @ grad, EPS)
                    hit_readback.soft_hit_distance[i_h, i_b] = d
            else:
                qd.atomic_or(errno[i_b], array_class.ErrorCode.OVERFLOW_MOCHI_CONTACTS)
        if qd.static(record):
            for i in qd.static(range(3)):
                qd.atomic_add(soft_state.verts_contact_force[tri[i], i_b], (w * bary[i]) * force)
            qd.atomic_add(dyn_state.links.contact_force[i_lb, i_b], -w * force)


@qd.kernel
def kernel_soft_contact_eval(
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    sdf_info: array_class.SDFInfo,
    mochi_info: MochiInfo,
    mochi_state: MochiState,
    soft_info: MochiSoftInfo,
    soft_state: MochiSoftState,
    hit_readback: MochiHitReadback,
    rigid_config: qd.template(),
    mochi_config: qd.template(),
    max_samples_per_entity: int,
    assem_res: qd.template(),
    assem_dres: qd.i32,
    skip_ls_done: qd.i32,
    record: qd.template(),
    errno: qd.Tensor,
):
    func_soft_contact_eval(
        0,
        False,
        mochi_state.all_envs,
        mochi_state.n_envs_all,
        dyn_state,
        dyn_info,
        sdf_info,
        mochi_info,
        mochi_state,
        soft_info,
        soft_state,
        hit_readback,
        rigid_config,
        mochi_config,
        max_samples_per_entity,
        assem_res,
        assem_dres,
        skip_ls_done,
        record,
        errno,
    )


@qd.func
def func_soft_pairs_to_blocks(
    i_b_env,
    per_env: qd.template(),
    envs: qd.types.ndarray(),
    n_envs: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    mochi_info: MochiInfo,
    mochi_state: MochiState,
    soft_state: MochiSoftState,
    rigid_config: qd.template(),
    assem_obj: qd.template(),
    assem_res: qd.template(),
    assem_dres,
    skip_ls_done,
):
    """Rigid side of the deformable-rigid pairs: residual and 6x6 block of the collider link (its point Jacobian is
    -[I, -[r_b]x], see the rigid kernel_pairs_to_blocks)."""
    max_pairs = soft_state.pair_entity_a.shape[0]
    _B = soft_state.verts_pos.shape[1]
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_p, i_slot in qd.ndrange(max_pairs, n_envs[None]) if qd.static(not per_env) else qd.ndrange(max_pairs, 1):
        i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
        if not func_is_env_active(i_b, mochi_state, skip_ls_done):
            continue
        if i_p >= soft_state.n_pairs[i_b] or soft_state.n_hits[i_p, i_b] == 0:
            continue
        i_lb = soft_state.pair_link_b[i_p, i_b]
        if qd.static(assem_obj):
            qd.atomic_add(mochi_state.obj[i_b], soft_state.acc_obj[i_p, i_b])
        if not mochi_info.links.is_dynamic[i_lb]:
            continue
        if qd.static(assem_res):
            F = soft_state.acc_f[i_p, i_b]
            Q = soft_state.acc_q[i_p, i_b]
            for k in qd.static(range(3)):
                qd.atomic_add(mochi_state.links_res[i_lb, i_b][k], F[k])
                qd.atomic_add(mochi_state.links_res[i_lb, i_b][3 + k], Q[k])
        if assem_dres:
            Dbar = func_sym6_to_mat3(soft_state.acc_D[i_p, i_b])
            Sh = soft_state.acc_SD[i_p, i_b]
            Sh2 = func_sym6_to_mat3(soft_state.acc_SDS[i_p, i_b])
            ShT = Sh.transpose()
            for k in qd.static(range(3)):
                for l in qd.static(range(3)):
                    qd.atomic_add(mochi_state.H_diag[i_lb, i_b][k, l], Dbar[k, l])
                    qd.atomic_add(mochi_state.H_diag[i_lb, i_b][k, 3 + l], ShT[k, l])
                    qd.atomic_add(mochi_state.H_diag[i_lb, i_b][3 + k, l], Sh[k, l])
                    qd.atomic_add(mochi_state.H_diag[i_lb, i_b][3 + k, 3 + l], -Sh2[k, l])


@qd.kernel
def kernel_soft_pairs_to_blocks(
    dyn_state: array_class.DynState,
    mochi_info: MochiInfo,
    mochi_state: MochiState,
    soft_state: MochiSoftState,
    rigid_config: qd.template(),
    assem_obj: qd.template(),
    assem_res: qd.template(),
    assem_dres: qd.i32,
    skip_ls_done: qd.i32,
):
    func_soft_pairs_to_blocks(
        0,
        False,
        mochi_state.all_envs,
        mochi_state.n_envs_all,
        dyn_state,
        mochi_info,
        mochi_state,
        soft_state,
        rigid_config,
        assem_obj,
        assem_res,
        assem_dres,
        skip_ls_done,
    )


# ------------------------------------------------------------------------------------
# ----------------------------------- linear algebra ---------------------------------
# ------------------------------------------------------------------------------------


@qd.func
def func_soft_point_displacement(
    i_lb,
    i_b,
    r_b,
    src: qd.Tensor,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_config: qd.template(),
):
    """Displacement of the collider point (lever arm r_b about the link origin) per the link's ancestor increments."""
    v6 = func_jacobian_times_dofs(i_lb, i_b, src, dyn_state, dyn_info, rigid_config)
    vel = qd.Vector([v6[0], v6[1], v6[2]], dt=gs.qd_float)
    ang = qd.Vector([v6[3], v6[4], v6[5]], dt=gs.qd_float)
    return vel + ang.cross(r_b)


@qd.func
def func_soft_point_force_add(
    i_lb,
    i_b,
    r_b,
    g,
    dst: qd.Tensor,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_config: qd.template(),
):
    """dst += J_b^T (g, r_b x g): a force g at the collider point as a wrench on the link's ancestor degrees of
    freedom."""
    torque = r_b.cross(g)
    func_jacobian_transpose_add(
        i_lb,
        i_b,
        qd.Vector([g[0], g[1], g[2], torque[0], torque[1], torque[2]], dt=gs.qd_float),
        dst,
        dyn_state,
        dyn_info,
        rigid_config,
    )


@qd.func
def func_soft_hit_counts_max(
    i_b_env,
    per_env: qd.template(),
    envs: qd.types.ndarray(),
    n_envs: qd.types.ndarray(),
    soft_state: MochiSoftState,
    rigid_config: qd.template(),
):
    """Largest per-environment hit counts of the current assembly, the runtime bounds of the loops over the compact hit
    lists (their capacities are many times larger than what a step records)."""
    _B = soft_state.verts_pos.shape[1]
    soft_state.n_soft_hits_max[None] = 0
    soft_state.n_sc_hits_max[None] = 0
    soft_state.n_pc_hits_max[None] = 0
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.ALL))
    for i_slot in range(n_envs[None]) if qd.static(not per_env) else range(1):
        i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
        qd.atomic_max(soft_state.n_soft_hits_max[None], soft_state.n_soft_hits[i_b])
        qd.atomic_max(soft_state.n_sc_hits_max[None], soft_state.n_sc_hits[i_b])
        qd.atomic_max(soft_state.n_pc_hits_max[None], soft_state.n_pc_hits[i_b])


@qd.func
def func_attachments_stage_start(
    i_b_env,
    per_env: qd.template(),
    envs: qd.types.ndarray(),
    n_envs: qd.types.ndarray(),
    mochi_state: MochiState,
    soft_info: MochiSoftInfo,
    soft_state: MochiSoftState,
    rigid_config: qd.template(),
):
    """Attachment violations at the stage start, the reference of the penalty damping. Runs after the stage-start
    poses of the links are stored."""
    n_att = soft_info.att_vert.shape[0]
    _B = mochi_state.is_active.shape[0]
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_a, i_slot in qd.ndrange(n_att, n_envs[None]) if qd.static(not per_env) else qd.ndrange(n_att, 1):
        i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
        if i_a >= soft_info.n_attachments[None]:
            continue
        i_v = soft_info.att_vert[i_a]
        i_l = soft_info.att_link[i_a]
        rho = gu.qd_transform_by_quat(soft_info.att_pos_local[i_a], mochi_state.links_quat_stage_start[i_l, i_b])
        soft_state.att_c_start[i_a, i_b] = (
            soft_state.verts_pos_stage_start[i_v, i_b] - mochi_state.links_pos_stage_start[i_l, i_b] - rho
        )


@qd.kernel
def kernel_attachments_stage_start(
    mochi_state: MochiState,
    soft_info: MochiSoftInfo,
    soft_state: MochiSoftState,
    rigid_config: qd.template(),
):
    func_attachments_stage_start(
        0, False, mochi_state.all_envs, mochi_state.n_envs_all, mochi_state, soft_info, soft_state, rigid_config
    )


@qd.func
def func_assemble_attachments(
    i_b_env,
    per_env: qd.template(),
    envs: qd.types.ndarray(),
    n_envs: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    mochi_state: MochiState,
    soft_info: MochiSoftInfo,
    soft_state: MochiSoftState,
    rigid_config: qd.template(),
    assem_obj: qd.template(),
    assem_res: qd.template(),
    assem_dres,
    skip_ls_done,
):
    """Penalty of every rigid-deformable attachment: E = 1/2 k |c|^2 + 1/2 (d / h) |c - c_stage_start|^2 on the
    violation c = x_v - (t_l + R_l p_local), with Gauss-Newton blocks on the vertex and on the link; the coupling
    between them is applied by the linear solver straight from the attachment tables."""
    n_att = soft_info.att_vert.shape[0]
    _B = mochi_state.is_active.shape[0]
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_a, i_slot in qd.ndrange(n_att, n_envs[None]) if qd.static(not per_env) else qd.ndrange(n_att, 1):
        i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
        if not func_is_env_active(i_b, mochi_state, skip_ls_done) or i_a >= soft_info.n_attachments[None]:
            continue
        i_v = soft_info.att_vert[i_a]
        i_l = soft_info.att_link[i_a]
        k = soft_info.att_stiffness[i_a]
        kappa = soft_info.att_damping[i_a] / mochi_state.dt_stage[i_b]
        K = k + kappa
        rho = gu.qd_transform_by_quat(soft_info.att_pos_local[i_a], dyn_state.links.quat[i_l, i_b])
        c = soft_state.verts_pos[i_v, i_b] - dyn_state.links.pos[i_l, i_b] - rho
        dc = c - soft_state.att_c_start[i_a, i_b]
        g = k * c + kappa * dc
        is_dynamic = soft_info.att_link_is_dynamic[i_a] != 0
        if qd.static(assem_obj):
            qd.atomic_add(mochi_state.obj[i_b], 0.5 * k * c.dot(c) + 0.5 * kappa * dc.dot(dc))
        if qd.static(assem_res):
            func_add_soft_vec(mochi_state.res, i_v, i_b, g, soft_info)
            if is_dynamic:
                torque = rho.cross(g)
                for kk in qd.static(range(3)):
                    qd.atomic_add(mochi_state.links_res[i_l, i_b][kk], -g[kk])
                    qd.atomic_add(mochi_state.links_res[i_l, i_b][3 + kk], -torque[kk])
        if assem_dres:
            qd.atomic_add(soft_state.verts_H_diag[i_v, i_b], K * qd.Matrix.identity(gs.qd_float, 3))
            if is_dynamic:
                S = skew(rho)
                SS = -K * (S @ S)
                I3 = qd.Matrix.identity(gs.qd_float, 3)
                for kk, ll in qd.static(qd.ndrange(3, 3)):
                    qd.atomic_add(mochi_state.H_diag[i_l, i_b][kk, ll], K * I3[kk, ll])
                    qd.atomic_add(mochi_state.H_diag[i_l, i_b][kk, 3 + ll], -K * S[kk, ll])
                    qd.atomic_add(mochi_state.H_diag[i_l, i_b][3 + kk, ll], K * S[kk, ll])
                    qd.atomic_add(mochi_state.H_diag[i_l, i_b][3 + kk, 3 + ll], SS[kk, ll])


@qd.kernel
def kernel_assemble_attachments(
    dyn_state: array_class.DynState,
    mochi_state: MochiState,
    soft_info: MochiSoftInfo,
    soft_state: MochiSoftState,
    rigid_config: qd.template(),
    assem_obj: qd.template(),
    assem_res: qd.template(),
    assem_dres: qd.i32,
    skip_ls_done: qd.i32,
):
    func_assemble_attachments(
        0,
        False,
        mochi_state.all_envs,
        mochi_state.n_envs_all,
        dyn_state,
        mochi_state,
        soft_info,
        soft_state,
        rigid_config,
        assem_obj,
        assem_res,
        assem_dres,
        skip_ls_done,
    )


@qd.func
def func_soft_matvec(
    i_b_env,
    per_env: qd.template(),
    envs: qd.types.ndarray(),
    n_envs: qd.types.ndarray(),
    src: qd.Tensor,
    dst: qd.Tensor,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    mochi_info: MochiInfo,
    mochi_state: MochiState,
    soft_info: MochiSoftInfo,
    soft_state: MochiSoftState,
    rigid_config: qd.template(),
):
    """dst += H_soft src for the running conjugate gradient environments: element blocks, per-sample vertex blocks
    b_i b_j D and the coupling with the collider links -b_i D J_b; fixed vertices act as identity rows."""
    n_verts = soft_state.verts_pos.shape[0]
    _B = soft_state.verts_pos.shape[1]

    n_soft_dofs = soft_info.csr_start.shape[0] - 1
    dof_start = soft_info.dof_start[None]
    twist_dof_start = soft_info.twist_dof_start[None]
    n_vert_rows = (twist_dof_start - dof_start) // 3
    # One pass covers the vertex rows and the rod twist rows. The three rows of a vertex share one column sequence
    # (the pattern is built from whole vertex blocks): one thread per vertex reads every column index and source
    # value once for the three rows; a twist row is a scalar walk of its own sequence.
    n_twist_rows = n_soft_dofs - 3 * n_vert_rows
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_x, i_slot in (
        qd.ndrange(n_vert_rows + n_twist_rows, n_envs[None])
        if qd.static(not per_env)
        else qd.ndrange(n_vert_rows + n_twist_rows, 1)
    ):
        i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
        if not mochi_state.pcg_is_active[i_b]:
            continue
        if i_x < n_vert_rows:
            i_v = i_x
            if not soft_state.verts_is_fixed[i_v, i_b]:
                j0 = soft_info.csr_start[3 * i_v]
                j1 = soft_info.csr_start[3 * i_v + 1]
                j2 = soft_info.csr_start[3 * i_v + 2]
                acc = qd.Vector.zero(gs.qd_float, 3)
                for jj in range(j1 - j0):
                    j_l = soft_info.csr_col[j0 + jj]
                    j_d = dof_start + j_l
                    if j_d < twist_dof_start and soft_state.verts_is_fixed[j_l // 3, i_b]:
                        continue
                    x = src[j_d, i_b]
                    acc[0] += soft_state.csr_values[j0 + jj, i_b] * x
                    acc[1] += soft_state.csr_values[j1 + jj, i_b] * x
                    acc[2] += soft_state.csr_values[j2 + jj, i_b] * x
                for k in qd.static(range(3)):
                    dst[dof_start + 3 * i_v + k, i_b] += acc[k]
        else:
            i_l = 3 * n_vert_rows + (i_x - n_vert_rows)
            acc_t = gs.qd_float(0.0)
            for j in range(soft_info.csr_start[i_l], soft_info.csr_start[i_l + 1]):
                j_l = soft_info.csr_col[j]
                j_d = dof_start + j_l
                if j_d < twist_dof_start and soft_state.verts_is_fixed[j_l // 3, i_b]:
                    continue
                acc_t += soft_state.csr_values[j, i_b] * src[j_d, i_b]
            dst[dof_start + i_l, i_b] += acc_t

    # One segmented pass covers the contact hits of the three kinds, the attachments and the fixed-row identities:
    # they are independent accumulations into dst (nothing writes a fixed vertex's rows but the identity), and each
    # offloaded loop costs a fixed launch on the device.
    n_h_soft = soft_state.n_soft_hits_max[None]
    n_h_sc = soft_state.n_sc_hits_max[None]
    n_h_pc = soft_state.n_pc_hits_max[None]
    n_att = soft_info.att_vert.shape[0]
    n_soft_items = n_h_soft + n_h_sc + n_h_pc + n_att + n_verts
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_x, i_slot in (
        qd.ndrange(n_soft_items, n_envs[None]) if qd.static(not per_env) else qd.ndrange(n_soft_items, 1)
    ):
        i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
        if not mochi_state.pcg_is_active[i_b]:
            continue
        if i_x < n_h_soft:
            i_h = i_x
            if i_h < soft_state.n_soft_hits[i_b]:
                i_s = soft_state.hit_sample[i_h, i_b]
                tri = soft_info.samples_tri[i_s]
                bary = soft_info.samples_bary[i_s]
                D = func_sym6_to_mat3(soft_state.hit_D[i_h, i_b])
                i_lb = soft_state.hit_link_b[i_h, i_b]
                # Relative displacement of the sample against the collider point.
                dp = qd.Vector.zero(gs.qd_float, 3)
                for i in qd.static(range(3)):
                    if not soft_state.verts_is_fixed[tri[i], i_b]:
                        dp += bary[i] * func_read_soft_vec(src, tri[i], i_b, soft_info)
                r_b = soft_state.hit_r_b[i_h, i_b]
                if i_lb >= 0:
                    dp -= func_soft_point_displacement(i_lb, i_b, r_b, src, dyn_state, dyn_info, rigid_config)
                g = D @ dp
                for i in qd.static(range(3)):
                    if not soft_state.verts_is_fixed[tri[i], i_b]:
                        func_add_soft_vec(dst, tri[i], i_b, bary[i] * g, soft_info)
                if i_lb >= 0:
                    # The rigid-rigid part J_b^T D J_b is already in the link block; only the coupling remains.
                    g_soft = qd.Vector.zero(gs.qd_float, 3)
                    for i in qd.static(range(3)):
                        if not soft_state.verts_is_fixed[tri[i], i_b]:
                            g_soft += bary[i] * func_read_soft_vec(src, tri[i], i_b, soft_info)
                    func_soft_point_force_add(i_lb, i_b, r_b, -(D @ g_soft), dst, dyn_state, dyn_info, rigid_config)
        elif i_x < n_h_soft + n_h_sc:
            i_h = i_x - n_h_soft
            if i_h < soft_state.n_sc_hits[i_b]:
                D = func_sym6_to_mat3(soft_state.sc_hit_D[i_h, i_b])
                kind_a = soft_state.sc_hit_kind_a[i_h, i_b]
                i_la = soft_state.sc_hit_link_a[i_h, i_b]
                r_a = soft_state.sc_hit_r_a[i_h, i_b]
                v_b = soft_info.elems_v[soft_state.sc_hit_elem_b[i_h, i_b]]
                bary_b = soft_state.sc_hit_bary_b[i_h, i_b]
                # Relative displacement of the colliding point against the collider point (barycentric in its
                # tetrahedron).
                dp = qd.Vector.zero(gs.qd_float, 3)
                tri_a = soft_info.samples_tri[soft_state.sc_hit_sample_a[i_h, i_b]]
                bary_a = soft_info.samples_bary[soft_state.sc_hit_sample_a[i_h, i_b]]
                if kind_a == 1:
                    for i in qd.static(range(3)):
                        if not soft_state.verts_is_fixed[tri_a[i], i_b]:
                            dp += bary_a[i] * func_read_soft_vec(src, tri_a[i], i_b, soft_info)
                elif i_la >= 0:
                    dp += func_soft_point_displacement(i_la, i_b, r_a, src, dyn_state, dyn_info, rigid_config)
                dp_b = qd.Vector.zero(gs.qd_float, 3)
                for j in qd.static(range(4)):
                    if not soft_state.verts_is_fixed[v_b[j], i_b]:
                        dp_b += bary_b[j] * func_read_soft_vec(src, v_b[j], i_b, soft_info)
                g = D @ (dp - dp_b)
                if kind_a == 1:
                    for i in qd.static(range(3)):
                        if not soft_state.verts_is_fixed[tri_a[i], i_b]:
                            func_add_soft_vec(dst, tri_a[i], i_b, bary_a[i] * g, soft_info)
                elif i_la >= 0:
                    # The rigid-rigid part J_a^T D J_a is already in the link block; only the coupling remains.
                    func_soft_point_force_add(i_la, i_b, r_a, -(D @ dp_b), dst, dyn_state, dyn_info, rigid_config)
                for j in qd.static(range(4)):
                    if not soft_state.verts_is_fixed[v_b[j], i_b]:
                        func_add_soft_vec(dst, v_b[j], i_b, -bary_b[j] * g, soft_info)
        elif i_x < n_h_soft + n_h_sc + n_h_pc:
            i_h = i_x - n_h_soft - n_h_sc
            if i_h < soft_state.n_pc_hits[i_b]:
                D = func_sym6_to_mat3(soft_state.pc_hit_D[i_h, i_b])
                kind_a = soft_state.pc_hit_kind_a[i_h, i_b]
                i_la = soft_state.pc_hit_link_a[i_h, i_b]
                r_a = soft_state.pc_hit_r_a[i_h, i_b]
                i_vb = soft_state.pc_hit_vert_b[i_h, i_b]
                tri_a = soft_info.samples_tri[soft_state.pc_hit_sample_a[i_h, i_b]]
                bary_a = soft_info.samples_bary[soft_state.pc_hit_sample_a[i_h, i_b]]
                dp = qd.Vector.zero(gs.qd_float, 3)
                if kind_a == 1:
                    for i in qd.static(range(3)):
                        if not soft_state.verts_is_fixed[tri_a[i], i_b]:
                            dp += bary_a[i] * func_read_soft_vec(src, tri_a[i], i_b, soft_info)
                elif i_la >= 0:
                    dp += func_soft_point_displacement(i_la, i_b, r_a, src, dyn_state, dyn_info, rigid_config)
                dp_b = qd.Vector.zero(gs.qd_float, 3)
                if not soft_state.verts_is_fixed[i_vb, i_b]:
                    dp_b = func_read_soft_vec(src, i_vb, i_b, soft_info)
                g = D @ (dp - dp_b)
                if kind_a == 1:
                    for i in qd.static(range(3)):
                        if not soft_state.verts_is_fixed[tri_a[i], i_b]:
                            func_add_soft_vec(dst, tri_a[i], i_b, bary_a[i] * g, soft_info)
                elif i_la >= 0:
                    func_soft_point_force_add(i_la, i_b, r_a, -(D @ dp_b), dst, dyn_state, dyn_info, rigid_config)
                if not soft_state.verts_is_fixed[i_vb, i_b]:
                    func_add_soft_vec(dst, i_vb, i_b, -g, soft_info)
        elif i_x < n_h_soft + n_h_sc + n_h_pc + n_att:
            i_a = i_x - n_h_soft - n_h_sc - n_h_pc
            if i_a < soft_info.n_attachments[None]:
                i_v = soft_info.att_vert[i_a]
                i_l = soft_info.att_link[i_a]
                K = soft_info.att_stiffness[i_a] + soft_info.att_damping[i_a] / mochi_state.dt_stage[i_b]
                is_fixed = soft_state.verts_is_fixed[i_v, i_b]
                dp = qd.Vector.zero(gs.qd_float, 3)
                if not is_fixed:
                    dp = func_read_soft_vec(src, i_v, i_b, soft_info)
                if soft_info.att_link_is_dynamic[i_a] != 0:
                    rho = gu.qd_transform_by_quat(soft_info.att_pos_local[i_a], dyn_state.links.quat[i_l, i_b])
                    g_soft = K * dp
                    dp -= func_soft_point_displacement(i_l, i_b, rho, src, dyn_state, dyn_info, rigid_config)
                    # The rigid-rigid part J^T K J is already in the link block; only the coupling remains.
                    func_soft_point_force_add(i_l, i_b, rho, -g_soft, dst, dyn_state, dyn_info, rigid_config)
                if not is_fixed:
                    func_add_soft_vec(dst, i_v, i_b, K * dp, soft_info)
        else:
            i_v = i_x - n_h_soft - n_h_sc - n_h_pc - n_att
            if soft_state.verts_is_fixed[i_v, i_b]:
                for k in qd.static(range(3)):
                    i_d = func_soft_dof(i_v, k, soft_info)
                    dst[i_d, i_b] = src[i_d, i_b]


@qd.func
def func_soft_precondition(
    i_b_env,
    per_env: qd.template(),
    envs: qd.types.ndarray(),
    n_envs: qd.types.ndarray(),
    r: qd.Tensor,
    z: qd.Tensor,
    mochi_state: MochiState,
    soft_info: MochiSoftInfo,
    soft_state: MochiSoftState,
    rigid_config: qd.template(),
    eps,
):
    """z = M^-1 r: scalar Jacobi on the rigid degrees of freedom, the twist diagonal on closed-loop rod twists and
    the 3x3 block Jacobi on the vertex blocks, in one segmented pass; the per-rod band solve follows."""
    n_verts = soft_state.verts_pos.shape[0]
    _B = soft_state.verts_pos.shape[1]
    n_rod_elems = soft_state.rod_elems_H.shape[0]
    # The deformable dofs are all written by the deformable branches below (vertex blocks, rod band, twist
    # diagonal): the scalar Jacobi branch covers the rigid dofs only.
    n_jacobi = soft_info.dof_start[None]
    n_items = n_jacobi + n_rod_elems + n_verts
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_x, i_slot in qd.ndrange(n_items, n_envs[None]) if qd.static(not per_env) else qd.ndrange(n_items, 1):
        i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
        if not mochi_state.pcg_is_active[i_b]:
            continue
        if i_x < n_jacobi:
            z[i_x, i_b] = r[i_x, i_b] / qd.max(mochi_state.pcg_diag[i_x, i_b], eps)
        elif i_x < n_jacobi + n_rod_elems:
            i_r = i_x - n_jacobi
            if soft_info.rod_elems_L[i_r] > 0.0:
                i_d = func_rod_twist_dof(i_r, soft_info)
                if soft_info.dofs_band_row[i_d] < 0:
                    diag = mochi_state.dofs_H_diag[i_d, i_b] + soft_state.rod_elems_twist_pcg[i_r, i_b]
                    z[i_d, i_b] = r[i_d, i_b] / qd.max(diag, eps)
        else:
            i_v = i_x - n_jacobi - n_rod_elems
            if soft_info.dofs_band_row[func_soft_dof(i_v, 0, soft_info)] < 0:
                r_v = func_read_soft_vec(r, i_v, i_b, soft_info)
                z_v = r_v
                if not soft_state.verts_is_fixed[i_v, i_b]:
                    H = soft_state.verts_H_diag[i_v, i_b]
                    det = H.determinant()
                    if det > eps:
                        z_v = H.inverse() @ r_v
                    else:
                        for k in qd.static(range(3)):
                            z_v[k] = r_v[k] / qd.max(H[k, k], eps)
                i_d = func_soft_dof(i_v, 0, soft_info)
                for k in qd.static(range(3)):
                    z[i_d + k, i_b] = z_v[k]
    func_rod_band_solve(i_b_env, per_env, envs, n_envs, r, z, mochi_state, soft_info, soft_state, rigid_config)


@qd.func
def func_soft_condense_dense(
    i_b_env,
    per_env: qd.template(),
    envs: qd.types.ndarray(),
    n_envs: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    mochi_state: MochiState,
    soft_info: MochiSoftInfo,
    soft_state: MochiSoftState,
    island_state: MochiIslandState,
    rigid_config: qd.template(),
):
    """Add the deformable blocks to the dense Hessian of every running environment and impose the Dirichlet rows and
    columns (zero off-diagonal, unit diagonal) of the fixed vertices."""
    n_verts = soft_state.verts_pos.shape[0]
    n_dofs = mochi_state.res.shape[0]
    _B = soft_state.verts_pos.shape[1]

    func_soft_hit_counts_max(i_b_env, per_env, envs, n_envs, soft_state, rigid_config)
    n_soft_dofs = soft_info.csr_start.shape[0] - 1
    dof_start = soft_info.dof_start[None]
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_l, i_slot in qd.ndrange(n_soft_dofs, n_envs[None]) if qd.static(not per_env) else qd.ndrange(n_soft_dofs, 1):
        i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
        if not (mochi_state.is_active[i_b] and island_state.uses_dense[i_b]):
            continue
        for j in range(soft_info.csr_start[i_l], soft_info.csr_start[i_l + 1]):
            mochi_state.H_dense[i_b, dof_start + i_l, dof_start + soft_info.csr_col[j]] += soft_state.csr_values[j, i_b]

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_h, i_slot in (
        qd.ndrange(soft_state.n_soft_hits_max[None], n_envs[None])
        if qd.static(not per_env)
        else qd.ndrange(soft_state.n_soft_hits_max[None], 1)
    ):
        i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
        if not (mochi_state.is_active[i_b] and island_state.uses_dense[i_b]) or i_h >= soft_state.n_soft_hits[i_b]:
            continue
        i_s = soft_state.hit_sample[i_h, i_b]
        tri = soft_info.samples_tri[i_s]
        bary = soft_info.samples_bary[i_s]
        D = func_sym6_to_mat3(soft_state.hit_D[i_h, i_b])
        for i in qd.static(range(3)):
            for j in qd.static(range(3)):
                i_d = func_soft_dof(tri[i], 0, soft_info)
                j_d = func_soft_dof(tri[j], 0, soft_info)
                for r in qd.static(range(3)):
                    for c in qd.static(range(3)):
                        qd.atomic_add(mochi_state.H_dense[i_b, i_d + r, j_d + c], bary[i] * bary[j] * D[r, c])
        i_lb = soft_state.hit_link_b[i_h, i_b]
        if i_lb >= 0:
            r_b = soft_state.hit_r_b[i_h, i_b]
            i_a = i_lb
            while i_a != -1:
                I_a = [i_a, i_b] if qd.static(rigid_config.batch_links_info) else i_a
                for k_d in range(dyn_info.links.dof_start[I_a], dyn_info.links.dof_end[I_a]):
                    vel, ang = func_link_dof_jacobian(i_lb, k_d, i_b, dyn_state)
                    column = -(D @ (vel + ang.cross(r_b)))
                    for i in qd.static(range(3)):
                        i_d = func_soft_dof(tri[i], 0, soft_info)
                        for r in qd.static(range(3)):
                            qd.atomic_add(mochi_state.H_dense[i_b, i_d + r, k_d], bary[i] * column[r])
                            qd.atomic_add(mochi_state.H_dense[i_b, k_d, i_d + r], bary[i] * column[r])
                i_a = dyn_info.links.parent_idx[I_a]

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_h, i_slot in (
        qd.ndrange(soft_state.n_sc_hits_max[None], n_envs[None])
        if qd.static(not per_env)
        else qd.ndrange(soft_state.n_sc_hits_max[None], 1)
    ):
        i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
        if not (mochi_state.is_active[i_b] and island_state.uses_dense[i_b]) or i_h >= soft_state.n_sc_hits[i_b]:
            continue
        D = func_sym6_to_mat3(soft_state.sc_hit_D[i_h, i_b])
        kind_a = soft_state.sc_hit_kind_a[i_h, i_b]
        v_b = soft_info.elems_v[soft_state.sc_hit_elem_b[i_h, i_b]]
        bary_b = soft_state.sc_hit_bary_b[i_h, i_b]
        for j in qd.static(range(4)):
            for l in qd.static(range(4)):
                j_d = func_soft_dof(v_b[j], 0, soft_info)
                l_d = func_soft_dof(v_b[l], 0, soft_info)
                for r in qd.static(range(3)):
                    for c in qd.static(range(3)):
                        qd.atomic_add(mochi_state.H_dense[i_b, j_d + r, l_d + c], bary_b[j] * bary_b[l] * D[r, c])
        if kind_a == 1:
            tri_a = soft_info.samples_tri[soft_state.sc_hit_sample_a[i_h, i_b]]
            bary_a = soft_info.samples_bary[soft_state.sc_hit_sample_a[i_h, i_b]]
            for i in qd.static(range(3)):
                i_d = func_soft_dof(tri_a[i], 0, soft_info)
                for k in qd.static(range(3)):
                    k_d = func_soft_dof(tri_a[k], 0, soft_info)
                    for r in qd.static(range(3)):
                        for c in qd.static(range(3)):
                            qd.atomic_add(mochi_state.H_dense[i_b, i_d + r, k_d + c], bary_a[i] * bary_a[k] * D[r, c])
                for j in qd.static(range(4)):
                    j_d = func_soft_dof(v_b[j], 0, soft_info)
                    for r in qd.static(range(3)):
                        for c in qd.static(range(3)):
                            qd.atomic_add(mochi_state.H_dense[i_b, i_d + r, j_d + c], -bary_a[i] * bary_b[j] * D[r, c])
                            qd.atomic_add(mochi_state.H_dense[i_b, j_d + c, i_d + r], -bary_a[i] * bary_b[j] * D[r, c])
        else:
            i_la = soft_state.sc_hit_link_a[i_h, i_b]
            if i_la >= 0:
                r_a = soft_state.sc_hit_r_a[i_h, i_b]
                i_anc = i_la
                while i_anc != -1:
                    I_anc = [i_anc, i_b] if qd.static(rigid_config.batch_links_info) else i_anc
                    for k_d in range(dyn_info.links.dof_start[I_anc], dyn_info.links.dof_end[I_anc]):
                        vel, ang = func_link_dof_jacobian(i_la, k_d, i_b, dyn_state)
                        column = -(D @ (vel + ang.cross(r_a)))
                        for j in qd.static(range(4)):
                            j_d = func_soft_dof(v_b[j], 0, soft_info)
                            for r in qd.static(range(3)):
                                qd.atomic_add(mochi_state.H_dense[i_b, j_d + r, k_d], bary_b[j] * column[r])
                                qd.atomic_add(mochi_state.H_dense[i_b, k_d, j_d + r], bary_b[j] * column[r])
                    i_anc = dyn_info.links.parent_idx[I_anc]

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_h, i_slot in (
        qd.ndrange(soft_state.n_pc_hits_max[None], n_envs[None])
        if qd.static(not per_env)
        else qd.ndrange(soft_state.n_pc_hits_max[None], 1)
    ):
        i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
        if not (mochi_state.is_active[i_b] and island_state.uses_dense[i_b]) or i_h >= soft_state.n_pc_hits[i_b]:
            continue
        D = func_sym6_to_mat3(soft_state.pc_hit_D[i_h, i_b])
        kind_a = soft_state.pc_hit_kind_a[i_h, i_b]
        i_vb = soft_state.pc_hit_vert_b[i_h, i_b]
        b_d = func_soft_dof(i_vb, 0, soft_info)
        for r in qd.static(range(3)):
            for c in qd.static(range(3)):
                qd.atomic_add(mochi_state.H_dense[i_b, b_d + r, b_d + c], D[r, c])
        if kind_a == 1:
            tri_a = soft_info.samples_tri[soft_state.pc_hit_sample_a[i_h, i_b]]
            bary_a = soft_info.samples_bary[soft_state.pc_hit_sample_a[i_h, i_b]]
            for i in qd.static(range(3)):
                i_d = func_soft_dof(tri_a[i], 0, soft_info)
                for k in qd.static(range(3)):
                    k_d = func_soft_dof(tri_a[k], 0, soft_info)
                    for r in qd.static(range(3)):
                        for c in qd.static(range(3)):
                            qd.atomic_add(mochi_state.H_dense[i_b, i_d + r, k_d + c], bary_a[i] * bary_a[k] * D[r, c])
                for r in qd.static(range(3)):
                    for c in qd.static(range(3)):
                        qd.atomic_add(mochi_state.H_dense[i_b, i_d + r, b_d + c], -bary_a[i] * D[r, c])
                        qd.atomic_add(mochi_state.H_dense[i_b, b_d + c, i_d + r], -bary_a[i] * D[r, c])
        else:
            i_la = soft_state.pc_hit_link_a[i_h, i_b]
            if i_la >= 0:
                r_a = soft_state.pc_hit_r_a[i_h, i_b]
                i_anc = i_la
                while i_anc != -1:
                    I_anc = [i_anc, i_b] if qd.static(rigid_config.batch_links_info) else i_anc
                    for k_d in range(dyn_info.links.dof_start[I_anc], dyn_info.links.dof_end[I_anc]):
                        vel, ang = func_link_dof_jacobian(i_la, k_d, i_b, dyn_state)
                        column = -(D @ (vel + ang.cross(r_a)))
                        for r in qd.static(range(3)):
                            qd.atomic_add(mochi_state.H_dense[i_b, b_d + r, k_d], column[r])
                            qd.atomic_add(mochi_state.H_dense[i_b, k_d, b_d + r], column[r])
                    i_anc = dyn_info.links.parent_idx[I_anc]

    n_att = soft_info.att_vert.shape[0]
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_a, i_slot in qd.ndrange(n_att, n_envs[None]) if qd.static(not per_env) else qd.ndrange(n_att, 1):
        i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
        if not (mochi_state.is_active[i_b] and island_state.uses_dense[i_b]) or i_a >= soft_info.n_attachments[None]:
            continue
        i_v = soft_info.att_vert[i_a]
        if soft_state.verts_is_fixed[i_v, i_b]:
            continue
        K = soft_info.att_stiffness[i_a] + soft_info.att_damping[i_a] / mochi_state.dt_stage[i_b]
        b_d = func_soft_dof(i_v, 0, soft_info)
        for r in qd.static(range(3)):
            qd.atomic_add(mochi_state.H_dense[i_b, b_d + r, b_d + r], K)
        if soft_info.att_link_is_dynamic[i_a] != 0:
            i_la = soft_info.att_link[i_a]
            rho = gu.qd_transform_by_quat(soft_info.att_pos_local[i_a], dyn_state.links.quat[i_la, i_b])
            i_anc = i_la
            while i_anc != -1:
                I_anc = [i_anc, i_b] if qd.static(rigid_config.batch_links_info) else i_anc
                for k_d in range(dyn_info.links.dof_start[I_anc], dyn_info.links.dof_end[I_anc]):
                    vel, ang = func_link_dof_jacobian(i_la, k_d, i_b, dyn_state)
                    column = -K * (vel + ang.cross(rho))
                    for r in qd.static(range(3)):
                        qd.atomic_add(mochi_state.H_dense[i_b, b_d + r, k_d], column[r])
                        qd.atomic_add(mochi_state.H_dense[i_b, k_d, b_d + r], column[r])
                i_anc = dyn_info.links.parent_idx[I_anc]

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_v, i_slot in qd.ndrange(n_verts, n_envs[None]) if qd.static(not per_env) else qd.ndrange(n_verts, 1):
        i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
        if not (mochi_state.is_active[i_b] and island_state.uses_dense[i_b]) or not soft_state.verts_is_fixed[i_v, i_b]:
            continue
        for k in qd.static(range(3)):
            i_d = func_soft_dof(i_v, k, soft_info)
            for j_d in range(n_dofs):
                mochi_state.H_dense[i_b, i_d, j_d] = 0.0
                mochi_state.H_dense[i_b, j_d, i_d] = 0.0
            mochi_state.H_dense[i_b, i_d, i_d] = 1.0


@qd.kernel
def kernel_soft_condense_dense(
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    mochi_state: MochiState,
    soft_info: MochiSoftInfo,
    soft_state: MochiSoftState,
    island_state: MochiIslandState,
    rigid_config: qd.template(),
):
    func_soft_condense_dense(
        0,
        False,
        mochi_state.all_envs,
        mochi_state.n_envs_all,
        dyn_state,
        dyn_info,
        mochi_state,
        soft_info,
        soft_state,
        island_state,
        rigid_config,
    )


# ------------------------------------------------------------------------------------
# --------------------------------------- io -----------------------------------------
# ------------------------------------------------------------------------------------


@qd.kernel
def kernel_soft_get_state(
    pos: qd.types.ndarray(),
    vel: qd.types.ndarray(),
    pos_prev: qd.types.ndarray(),
    vel_prev: qd.types.ndarray(),
    soft_state: MochiSoftState,
    rigid_config: qd.template(),
):
    n_verts = soft_state.verts_pos.shape[0]
    _B = soft_state.verts_pos.shape[1]
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_v, i_b in qd.ndrange(n_verts, _B):
        for k in qd.static(range(3)):
            pos[i_b, i_v, k] = soft_state.verts_pos[i_v, i_b][k]
            vel[i_b, i_v, k] = soft_state.verts_vel[i_v, i_b][k]
            for j in qd.static(range(N_HISTORY)):
                pos_prev[i_b, j, i_v, k] = soft_state.verts_pos_prev[j, i_v, i_b][k]
                vel_prev[i_b, j, i_v, k] = soft_state.verts_vel_prev[j, i_v, i_b][k]


@qd.kernel
def kernel_soft_set_state(
    envs_idx: qd.types.ndarray(),
    pos: qd.types.ndarray(),
    vel: qd.types.ndarray(),
    pos_prev: qd.types.ndarray(),
    vel_prev: qd.types.ndarray(),
    soft_state: MochiSoftState,
    rigid_config: qd.template(),
):
    n_verts = soft_state.verts_pos.shape[0]
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_v, i_b_ in qd.ndrange(n_verts, envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        for k in qd.static(range(3)):
            soft_state.verts_pos[i_v, i_b][k] = pos[i_b, i_v, k]
            soft_state.verts_vel[i_v, i_b][k] = vel[i_b, i_v, k]
            for j in qd.static(range(N_HISTORY)):
                soft_state.verts_pos_prev[j, i_v, i_b][k] = pos_prev[i_b, j, i_v, k]
                soft_state.verts_vel_prev[j, i_v, i_b][k] = vel_prev[i_b, j, i_v, k]


@qd.kernel
def kernel_soft_get_entity_state(
    v_start: qd.i32,
    pos: qd.types.ndarray(),
    vel: qd.types.ndarray(),
    soft_state: MochiSoftState,
    rigid_config: qd.template(),
):
    n_verts = pos.shape[1]
    _B = pos.shape[0]
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_v_, i_b in qd.ndrange(n_verts, _B):
        for k in qd.static(range(3)):
            pos[i_b, i_v_, k] = soft_state.verts_pos[v_start + i_v_, i_b][k]
            vel[i_b, i_v_, k] = soft_state.verts_vel[v_start + i_v_, i_b][k]


@qd.kernel
def kernel_soft_get_vertices_field(
    envs_idx: qd.types.ndarray(),
    v_start: qd.i32,
    out: qd.types.ndarray(),
    field: qd.Tensor,
    rigid_config: qd.template(),
):
    n_verts = out.shape[1]
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_v_, i_b_ in qd.ndrange(n_verts, envs_idx.shape[0]):
        for k in qd.static(range(3)):
            out[i_b_, i_v_, k] = field[v_start + i_v_, envs_idx[i_b_]][k]


@qd.kernel
def kernel_soft_set_vertices_positions(
    envs_idx: qd.types.ndarray(),
    v_start: qd.i32,
    pos: qd.types.ndarray(),
    soft_state: MochiSoftState,
    rigid_config: qd.template(),
):
    """Set vertex positions; their velocity history is reset (the next step starts from rest at the new positions
    unless velocities are set afterwards)."""
    n_verts = pos.shape[1]
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_v_, i_b_ in qd.ndrange(n_verts, envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        i_v = v_start + i_v_
        value = qd.Vector([pos[i_b_, i_v_, 0], pos[i_b_, i_v_, 1], pos[i_b_, i_v_, 2]], dt=gs.qd_float)
        soft_state.verts_pos[i_v, i_b] = value
        soft_state.verts_target[i_v, i_b] = value
        soft_state.verts_vel[i_v, i_b] = qd.Vector.zero(gs.qd_float, 3)
        for j in qd.static(range(N_HISTORY)):
            soft_state.verts_pos_prev[j, i_v, i_b] = value
            soft_state.verts_vel_prev[j, i_v, i_b] = qd.Vector.zero(gs.qd_float, 3)


@qd.kernel
def kernel_soft_set_vertices_velocities(
    envs_idx: qd.types.ndarray(),
    v_start: qd.i32,
    vel: qd.types.ndarray(),
    soft_state: MochiSoftState,
    rigid_config: qd.template(),
):
    n_verts = vel.shape[1]
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_v_, i_b_ in qd.ndrange(n_verts, envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        i_v = v_start + i_v_
        value = qd.Vector([vel[i_b_, i_v_, 0], vel[i_b_, i_v_, 1], vel[i_b_, i_v_, 2]], dt=gs.qd_float)
        soft_state.verts_vel[i_v, i_b] = value
        for j in qd.static(range(N_HISTORY)):
            soft_state.verts_vel_prev[j, i_v, i_b] = value


@qd.kernel
def kernel_soft_set_vertices_fixed(
    envs_idx: qd.types.ndarray(),
    verts_idx: qd.types.ndarray(),
    is_fixed: qd.i32,
    soft_state: MochiSoftState,
    rigid_config: qd.template(),
):
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_v_, i_b_ in qd.ndrange(verts_idx.shape[0], envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        i_v = verts_idx[i_v_]
        soft_state.verts_is_fixed[i_v, i_b] = is_fixed != 0
        if is_fixed != 0:
            soft_state.verts_target[i_v, i_b] = soft_state.verts_pos[i_v, i_b]
            soft_state.verts_vel[i_v, i_b] = qd.Vector.zero(gs.qd_float, 3)
            for j in qd.static(range(N_HISTORY)):
                soft_state.verts_vel_prev[j, i_v, i_b] = qd.Vector.zero(gs.qd_float, 3)


@qd.kernel
def kernel_soft_set_vertices_target(
    envs_idx: qd.types.ndarray(),
    verts_idx: qd.types.ndarray(),
    pos: qd.types.ndarray(),
    soft_state: MochiSoftState,
    rigid_config: qd.template(),
):
    """Prescribe the end-of-step positions of the given vertices (moving Dirichlet condition): the vertices become
    fixed and reach their target within the next step; their velocity history is kept."""
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_v_, i_b_ in qd.ndrange(verts_idx.shape[0], envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        i_v = verts_idx[i_v_]
        soft_state.verts_is_fixed[i_v, i_b] = True
        soft_state.verts_target[i_v, i_b] = qd.Vector(
            [pos[i_b_, i_v_, 0], pos[i_b_, i_v_, 1], pos[i_b_, i_v_, 2]], dt=gs.qd_float
        )


@qd.kernel
def kernel_soft_set_entity_contact_params(
    i_e: qd.i32,
    params: qd.types.ndarray(),
    soft_info: MochiSoftInfo,
):
    soft_info.entities_penalty_coefficient[i_e] = params[0]
    soft_info.entities_friction[i_e] = params[1]
    soft_info.entities_penalty_smoothing_half_distance[i_e] = params[2]
    soft_info.entities_penalty_threshold[i_e] = params[3]
    soft_info.entities_friction_falloff_vel[i_e] = params[4]
    soft_info.entities_viscous_friction[i_e] = params[5]
    soft_info.entities_normal_viscous_damping[i_e] = params[6]


@qd.kernel
def kernel_rod_get_state_render(
    vverts_render: qd.Tensor,
    rod_vverts_vvert: qd.Tensor,
    rod_vverts_node: qd.Tensor,
    rod_vverts_elem: qd.Tensor,
    rod_vverts_offset: qd.Tensor,
    envs_offset: qd.types.ndarray(),
    soft_info: MochiSoftInfo,
    soft_state: MochiSoftState,
    rigid_config: qd.template(),
):
    """Tube vertices around the rod centerlines: node position plus the cross-section offset (radius times cosine and
    sine of the ring angle) along the material axis of the segment and its binormal."""
    n_vverts = rod_vverts_node.shape[0]
    _B = soft_state.verts_pos.shape[1]
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_vv, i_b in qd.ndrange(n_vverts, _B):
        i_v = rod_vverts_node[i_vv]
        i_r = rod_vverts_elem[i_vv]
        offset = rod_vverts_offset[i_vv]
        axis = soft_state.rod_elems_axis[i_r, i_b]
        tangent = func_rod_tangent(i_r, i_b, soft_state.verts_pos, soft_info)
        pos = soft_state.verts_pos[i_v, i_b] + offset[0] * axis + offset[1] * tangent.cross(axis)
        for k in qd.static(range(3)):
            vverts_render[rod_vverts_vvert[i_vv], i_b][k] = qd.cast(pos[k] + envs_offset[i_b, k], qd.f32)


@qd.kernel
def kernel_rod_init_render(
    vverts: qd.types.ndarray(),
    nodes: qd.types.ndarray(),
    elems: qd.types.ndarray(),
    offsets: qd.types.ndarray(),
    rod_vverts_vvert: qd.Tensor,
    rod_vverts_node: qd.Tensor,
    rod_vverts_elem: qd.Tensor,
    rod_vverts_offset: qd.Tensor,
):
    for i_vv in range(nodes.shape[0]):
        rod_vverts_vvert[i_vv] = vverts[i_vv]
        rod_vverts_node[i_vv] = nodes[i_vv]
        rod_vverts_elem[i_vv] = elems[i_vv]
        for k in qd.static(range(2)):
            rod_vverts_offset[i_vv][k] = offsets[i_vv, k]


@qd.kernel
def kernel_soft_get_state_render(
    vverts_render: qd.Tensor,
    vverts_vert_idx: qd.Tensor,
    vverts_elem: qd.Tensor,
    vverts_bary: qd.Tensor,
    envs_offset: qd.types.ndarray(),
    soft_info: MochiSoftInfo,
    soft_state: MochiSoftState,
    rigid_config: qd.template(),
):
    n_vverts = vverts_vert_idx.shape[0]
    _B = soft_state.verts_pos.shape[1]
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_vv, i_b in qd.ndrange(n_vverts, _B):
        pos = qd.Vector.zero(gs.qd_float, 3)
        i_el = vverts_elem[i_vv]
        if i_el >= 0:
            # An embedded render vertex follows the barycentric combination of its element's nodes.
            v = soft_info.elems_v[i_el]
            bary = vverts_bary[i_vv]
            for k in qd.static(range(4)):
                pos += bary[k] * soft_state.verts_pos[v[k], i_b]
        else:
            pos = soft_state.verts_pos[vverts_vert_idx[i_vv], i_b]
        for k in qd.static(range(3)):
            vverts_render[i_vv, i_b][k] = qd.cast(pos[k] + envs_offset[i_b, k], qd.f32)


@qd.kernel
def kernel_soft_init_render(
    vert_idx: qd.types.ndarray(),
    elems_idx: qd.types.ndarray(),
    bary: qd.types.ndarray(),
    vverts_vert_idx: qd.Tensor,
    vverts_elem: qd.Tensor,
    vverts_bary: qd.Tensor,
):
    for i_vv in range(vert_idx.shape[0]):
        vverts_vert_idx[i_vv] = vert_idx[i_vv]
        vverts_elem[i_vv] = elems_idx[i_vv]
        for k in qd.static(range(4)):
            vverts_bary[i_vv][k] = bary[i_vv, k]


@qd.kernel
def kernel_soft_set_links_pair_enabled(
    entities_links_pair_enabled: qd.types.ndarray(),
    soft_info: MochiSoftInfo,
    rigid_config: qd.template(),
):
    n_entities = entities_links_pair_enabled.shape[0]
    n_links = entities_links_pair_enabled.shape[1]
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_e, i_l in qd.ndrange(n_entities, n_links):
        soft_info.entities_links_pair_enabled[i_e, i_l] = entities_links_pair_enabled[i_e, i_l]


# ------------------------------------------------------------------------------------
# ----------------------------- deformable bodies as colliders -----------------------
# ------------------------------------------------------------------------------------


@qd.func
def func_soft_sdf(i_e, p, soft_info: MochiSoftInfo):
    """Trilinear rest-shape signed distance of a deformable entity at the material point p, and its analytic gradient.
    Points outside the grid are reported as invalid."""
    origin = soft_info.entities_sdf_origin[i_e]
    cell = soft_info.entities_sdf_cell[i_e]
    res = soft_info.entities_sdf_res[i_e]
    start = soft_info.entities_sdf_start[i_e]
    u = (p - origin) / cell  # per-axis cell size
    is_valid = True
    i0 = qd.Vector.zero(gs.qd_int, 3)
    f = qd.Vector.zero(gs.qd_float, 3)
    for k in qd.static(range(3)):
        if u[k] < 0.0 or u[k] > gs.qd_float(res[k] - 1):
            is_valid = False
        i0[k] = qd.min(qd.max(qd.cast(qd.floor(u[k]), gs.qd_int), 0), res[k] - 2)
        f[k] = u[k] - gs.qd_float(i0[k])
    d = gs.qd_float(0.0)
    grad = qd.Vector.zero(gs.qd_float, 3)
    if is_valid:
        c = qd.Vector.zero(gs.qd_float, 8)
        for a, b, cc in qd.static(qd.ndrange(2, 2, 2)):
            idx = start + ((i0[0] + a) * res[1] + (i0[1] + b)) * res[2] + (i0[2] + cc)
            c[4 * a + 2 * b + cc] = soft_info.sdf_values[idx]
        # Interpolate along z, then y, then x.
        c00 = c[0] * (1.0 - f[2]) + c[1] * f[2]
        c01 = c[2] * (1.0 - f[2]) + c[3] * f[2]
        c10 = c[4] * (1.0 - f[2]) + c[5] * f[2]
        c11 = c[6] * (1.0 - f[2]) + c[7] * f[2]
        c0 = c00 * (1.0 - f[1]) + c01 * f[1]
        c1 = c10 * (1.0 - f[1]) + c11 * f[1]
        d = c0 * (1.0 - f[0]) + c1 * f[0]
        dz00 = c[1] - c[0]
        dz01 = c[3] - c[2]
        dz10 = c[5] - c[4]
        dz11 = c[7] - c[6]
        grad[0] = (c1 - c0) / cell[0]
        grad[1] = ((c01 - c00) * (1.0 - f[0]) + (c11 - c10) * f[0]) / cell[1]
        grad[2] = (
            (dz00 * (1.0 - f[1]) + dz01 * f[1]) * (1.0 - f[0]) + (dz10 * (1.0 - f[1]) + dz11 * f[1]) * f[0]
        ) / cell[2]
    return is_valid, d, grad


# ------------------------------------------------------------------------------------
# ------------------------- spatial hash of the deformable colliders -----------------
# ------------------------------------------------------------------------------------
# Both deformable collider kinds (spheres of the point-cloud colliders, deformed tetrahedra of the grid colliders)
# are located by a spatial hash rebuilt at every assembly, as in mochi: every item is inserted in the bins of the (at
# most 2 x 2 x 2) cells its bounds overlap and a query walks the chain of its own cell. With a cell at least as large
# as the largest item extent (the contact-range diameter of a sphere, the bounds of a tetrahedron) the item's cells
# hold every point it can touch, so the candidate set is a superset of the exact one and the contact response,
# evaluated per candidate, is unchanged. Two cells hashed to the same bin share a chain: an entry is kept when its
# own cell (the item's lowest cell plus its offset) is the query cell, so every item is visited at most once.

HASH_X = 73856093
HASH_Y = 19349663
HASH_Z = 83492791


@qd.func
def func_hash_cell(pos, inv_cell):
    """Integer cell coordinates of a point."""
    return qd.cast(qd.floor(pos * inv_cell), gs.qd_int)


@qd.func
def func_hash_bin(cell, mask):
    """Bin of a cell: three primes hashed into a power-of-two table."""
    h = qd.cast(cell[0], qd.u32) * qd.u32(HASH_X)
    h = h ^ (qd.cast(cell[1], qd.u32) * qd.u32(HASH_Y))
    h = h ^ (qd.cast(cell[2], qd.u32) * qd.u32(HASH_Z))
    return qd.cast(h & qd.cast(mask, qd.u32), gs.qd_int)


@qd.func
def func_cell_offset(k):
    """Offset of the k-th of the 2 x 2 x 2 cells an item's bounds may overlap, from its lowest cell."""
    return qd.Vector([k // 4, (k // 2) % 2, k % 2], dt=gs.qd_int)


@qd.func
def func_hash_insert(heads: qd.template(), nexts: qd.template(), i_item, cell_lo, cell_hi, mask, i_b):
    """Insert an item in every cell between its lowest and highest cells (at most two per axis): entry 8 * item + k."""
    for k in qd.static(range(8)):
        cell = cell_lo + func_cell_offset(k)
        if (cell <= cell_hi).all():
            entry = 8 * i_item + k
            nexts[entry, i_b] = qd.atomic_exchange(heads[func_hash_bin(cell, mask), i_b], entry)


@qd.func
def func_pc_hash_build(
    i_b_env,
    per_env: qd.template(),
    envs: qd.types.ndarray(),
    n_envs: qd.types.ndarray(),
    mochi_state: MochiState,
    soft_info: MochiSoftInfo,
    soft_state: MochiSoftState,
    rigid_config: qd.template(),
    skip_ls_done,
):
    """Insert the collider spheres of the shell and rod vertices (padded by their contact range) in the hash."""
    n_bins = soft_state.pc_hash_heads.shape[0]
    n_verts = soft_state.verts_pos.shape[0]
    inv_cell = 1.0 / soft_info.pc_hash_cell[None]
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_bin, i_slot in qd.ndrange(n_bins, n_envs[None]) if qd.static(not per_env) else qd.ndrange(n_bins, 1):
        i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
        if func_is_env_active(i_b, mochi_state, skip_ls_done):
            soft_state.pc_hash_heads[i_bin, i_b] = -1
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_v, i_slot in qd.ndrange(n_verts, n_envs[None]) if qd.static(not per_env) else qd.ndrange(n_verts, 1):
        i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
        if func_is_env_active(i_b, mochi_state, skip_ls_done):
            i_e = soft_info.verts_entity_idx[i_v]
            if (
                soft_info.entities_collider_type[i_e] == COLLIDER_TYPE.POINT_CLOUD
                and soft_info.verts_collider_weight[i_v] > 0.0
            ):
                pad = soft_info.entities_collider_radius[i_e] + soft_info.entities_penalty_threshold[i_e]
                pos = soft_state.verts_pos[i_v, i_b]
                cell_lo = func_hash_cell(pos - pad, inv_cell)
                cell_hi = func_hash_cell(pos + pad, inv_cell)
                func_hash_insert(
                    soft_state.pc_hash_heads, soft_state.pc_hash_next, i_v, cell_lo, cell_hi, n_bins - 1, i_b
                )


@qd.func
def func_tet_aabb(i_el, i_b, soft_info: MochiSoftInfo, soft_state: MochiSoftState):
    """Bounds of a deformed tetrahedron."""
    v = soft_info.elems_v[i_el]
    aabb_min = soft_state.verts_pos[v[0], i_b]
    aabb_max = aabb_min
    for j in qd.static(range(1, 4)):
        pos = soft_state.verts_pos[v[j], i_b]
        aabb_min = qd.min(aabb_min, pos)
        aabb_max = qd.max(aabb_max, pos)
    return aabb_min, aabb_max


@qd.func
def func_tet_tree_refit(
    i_b_env,
    per_env: qd.template(),
    envs: qd.types.ndarray(),
    n_envs: qd.types.ndarray(),
    mochi_state: MochiState,
    soft_info: MochiSoftInfo,
    soft_state: MochiSoftState,
    rigid_config: qd.template(),
    mochi_config: qd.template(),
    skip_ls_done,
):
    """Refit the bounds of the tetrahedron hierarchy to the deformed vertices, one level at a time from the leaves up
    (the children of inner node i are i + 1 and the escape node of i + 1)."""
    for i_level in qd.static(range(mochi_config.tet_tree_levels)):
        n_level = soft_info.tet_tree_level_start[i_level + 1] - soft_info.tet_tree_level_start[i_level]
        qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
        for k, i_slot in qd.ndrange(n_level, n_envs[None]) if qd.static(not per_env) else qd.ndrange(n_level, 1):
            i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
            if func_is_env_active(i_b, mochi_state, skip_ls_done):
                i_node = soft_info.tet_tree_level_nodes[soft_info.tet_tree_level_start[i_level] + k]
                aabb_min = qd.Vector([gs.qd_float(1e30)] * 3, dt=gs.qd_float)
                aabb_max = qd.Vector([gs.qd_float(-1e30)] * 3, dt=gs.qd_float)
                if soft_info.tet_tree_is_leaf[i_node] != 0:
                    first = soft_info.tet_tree_first[i_node]
                    for j in range(first, first + soft_info.tet_tree_count[i_node]):
                        tet_min, tet_max = func_tet_aabb(soft_info.tet_tree_elems[j], i_b, soft_info, soft_state)
                        aabb_min = qd.min(aabb_min, tet_min)
                        aabb_max = qd.max(aabb_max, tet_max)
                else:
                    i_left = i_node + 1
                    i_right = soft_info.tet_tree_escape[i_left]
                    aabb_min = qd.min(soft_state.tet_tree_min[i_left, i_b], soft_state.tet_tree_min[i_right, i_b])
                    aabb_max = qd.max(soft_state.tet_tree_max[i_left, i_b], soft_state.tet_tree_max[i_right, i_b])
                soft_state.tet_tree_min[i_node, i_b] = aabb_min
                soft_state.tet_tree_max[i_node, i_b] = aabb_max


@qd.func
def func_query_point(
    i_q,
    i_b,
    dyn_state: array_class.DynState,
    mochi_info: MochiInfo,
    mochi_state: MochiState,
    soft_info: MochiSoftInfo,
    soft_state: MochiSoftState,
    EPS,
):
    """Colliding side of a contact query: a rigid link sample (i_q below the number of rigid samples) or a deformable
    boundary sample. Returns kind (0 rigid, 1 deformable), link, entity, sample index, position, stage-start position,
    outward normal, quadrature weight, and the contact parameters of the colliding side."""
    n_rigid = soft_info.n_rigid_queries[None]
    kind_a = 0
    i_la = -1
    e_a = -1
    i_sample = i_q
    pos = qd.Vector.zero(gs.qd_float, 3)
    pos_start = qd.Vector.zero(gs.qd_float, 3)
    normal_a = qd.Vector.zero(gs.qd_float, 3)
    w = gs.qd_float(0.0)
    k_a = gs.qd_float(0.0)
    falloff_a = gs.qd_float(0.0)
    mu_a = gs.qd_float(0.0)
    c_visc_a = gs.qd_float(0.0)
    c_ndamp_a = gs.qd_float(0.0)
    if i_q < n_rigid:
        i_la = mochi_info.samples.link_idx[i_q]
        i_ga = mochi_info.samples.geom_idx[i_q]
        pos_a = dyn_state.links.pos[i_la, i_b]
        quat_a = dyn_state.links.quat[i_la, i_b]
        pos = gu.qd_transform_by_trans_quat(mochi_info.samples.pos[i_q], pos_a, quat_a)
        pos_start = gu.qd_transform_by_trans_quat(
            mochi_info.samples.pos[i_q],
            mochi_state.links_pos_stage_start[i_la, i_b],
            mochi_state.links_quat_stage_start[i_la, i_b],
        )
        normal_a = gu.qd_transform_by_quat(mochi_info.samples.normal[i_q], quat_a)
        w = mochi_info.samples.weight[i_q]
        k_a = mochi_info.geoms.penalty_coefficient[i_ga]
        falloff_a = mochi_info.geoms.friction_falloff_vel[i_ga]
        mu_a = mochi_info.geoms.friction[i_ga]
        c_visc_a = mochi_info.geoms.viscous_friction[i_ga]
        c_ndamp_a = mochi_info.geoms.normal_viscous_damping[i_ga]
    else:
        kind_a = 1
        i_sample = i_q - n_rigid
        e_a = soft_info.samples_entity_idx[i_sample]
        tri = soft_info.samples_tri[i_sample]
        bary = soft_info.samples_bary[i_sample]
        x0 = soft_state.verts_pos[tri[0], i_b]
        x1 = soft_state.verts_pos[tri[1], i_b]
        x2 = soft_state.verts_pos[tri[2], i_b]
        pos = bary[0] * x0 + bary[1] * x1 + bary[2] * x2
        pos_start = (
            bary[0] * soft_state.verts_pos_stage_start[tri[0], i_b]
            + bary[1] * soft_state.verts_pos_stage_start[tri[1], i_b]
            + bary[2] * soft_state.verts_pos_stage_start[tri[2], i_b]
        )
        normal_a = gu.qd_normalize((x1 - x0).cross(x2 - x0), EPS)
        w = soft_info.samples_weight[i_sample]
        k_a = soft_info.entities_penalty_coefficient[e_a]
        falloff_a = soft_info.entities_friction_falloff_vel[e_a]
        mu_a = soft_info.entities_friction[e_a]
        c_visc_a = soft_info.entities_viscous_friction[e_a]
        c_ndamp_a = soft_info.entities_normal_viscous_damping[e_a]
    return kind_a, i_la, e_a, i_sample, pos, pos_start, normal_a, w, k_a, falloff_a, mu_a, c_visc_a, c_ndamp_a


@qd.func
def func_soft_collider_eval(
    i_b_env,
    per_env: qd.template(),
    envs: qd.types.ndarray(),
    n_envs: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    mochi_info: MochiInfo,
    mochi_state: MochiState,
    soft_info: MochiSoftInfo,
    soft_state: MochiSoftState,
    hit_readback: MochiHitReadback,
    rigid_config: qd.template(),
    mochi_config: qd.template(),
    assem_obj: qd.template(),
    assem_res: qd.template(),
    assem_dres,
    skip_ls_done,
    record: qd.template(),
    errno: qd.Tensor,
):
    """Evaluate every sample point against the deformed tetrahedra of the collider entities whose bounds contain it,
    found by descending the tetrahedron hierarchy: samples inside a tetrahedron are pulled back to the rest shape of
    the collider entity, where its signed distance field gives the penetration; the response acts on the colliding
    sample and on the four vertices of the tetrahedron."""
    n_queries = soft_info.n_queries[None]
    n_nodes = soft_state.tet_tree_min.shape[0]
    _B = soft_state.verts_pos.shape[1]
    max_hits = soft_state.sc_hit_kind_a.shape[0]
    EPS = mochi_info.EPS[None]

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_q, i_slot in qd.ndrange(n_queries, n_envs[None]) if qd.static(not per_env) else qd.ndrange(n_queries, 1):
        i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
        if not func_is_env_active(i_b, mochi_state, skip_ls_done):
            continue
        # A deformable sample whose entity has no tetrahedral collider to hit never queries; a static body is only a
        # collider: its samples never collide (mochi's rule).
        n_rigid = soft_info.n_rigid_queries[None]
        if i_q >= n_rigid and soft_info.entities_queries_tets[soft_info.samples_entity_idx[i_q - n_rigid]] == 0:
            continue
        if i_q < n_rigid and not mochi_info.links.is_dynamic[mochi_info.samples.link_idx[i_q]]:
            continue
        kind_a, i_la, e_a, i_sample, pos, pos_start, normal_a0, w, k_a, falloff_a, mu_a, c_visc_a, c_ndamp_a = (
            func_query_point(i_q, i_b, dyn_state, mochi_info, mochi_state, soft_info, soft_state, EPS)
        )
        # Descend the hierarchy: a node whose deformed bounds do not contain the sample is skipped with its subtree.
        brute_force = soft_info.tet_tree_brute_force[None] != 0
        i_node = 0
        while i_node < n_nodes:
            if not brute_force and (
                (pos < soft_state.tet_tree_min[i_node, i_b]).any() or (pos > soft_state.tet_tree_max[i_node, i_b]).any()
            ):
                i_node = soft_info.tet_tree_escape[i_node]
            else:
                if soft_info.tet_tree_is_leaf[i_node] != 0:
                    first = soft_info.tet_tree_first[i_node]
                    for j_leaf in range(first, first + soft_info.tet_tree_count[i_node]):
                        i_el_cur = soft_info.tet_tree_elems[j_leaf]
                        e_b = soft_info.elems_entity_idx[i_el_cur]
                        is_enabled = True
                        if kind_a == 0:
                            is_enabled = soft_info.entities_links_pair_enabled[e_b, i_la]
                        else:
                            is_enabled = (e_a != e_b) and soft_info.entities_pair_enabled[e_a, e_b]
                        if not is_enabled:
                            continue
                        # the sample must lie in the bounds of the tetrahedron before the inclusion test
                        v_cur = soft_info.elems_v[i_el_cur]
                        aabb_min = soft_state.verts_pos[v_cur[0], i_b]
                        aabb_max = aabb_min
                        for j in qd.static(range(1, 4)):
                            pos_j = soft_state.verts_pos[v_cur[j], i_b]
                            aabb_min = qd.min(aabb_min, pos_j)
                            aabb_max = qd.max(aabb_max, pos_j)
                        if (pos < aabb_min).any() or (pos > aabb_max).any():
                            continue
                        normal_a = normal_a0

                        # Inclusion in the deformed tetrahedron and pull-back to the rest shape.
                        v = soft_info.elems_v[i_el_cur]
                        x3 = soft_state.verts_pos[v[3], i_b]
                        Ds = qd.Matrix.cols(
                            [
                                soft_state.verts_pos[v[0], i_b] - x3,
                                soft_state.verts_pos[v[1], i_b] - x3,
                                soft_state.verts_pos[v[2], i_b] - x3,
                            ]
                        )
                        if qd.abs(Ds.determinant()) <= EPS:
                            continue
                        Ds_inv = Ds.inverse()
                        b3 = Ds_inv @ (pos - x3)
                        bary_b = qd.Vector([b3[0], b3[1], b3[2], 1.0 - b3[0] - b3[1] - b3[2]], dt=gs.qd_float)
                        if (bary_b < 0.0).any():
                            continue
                        Dm = soft_info.elems_Dm[i_el_cur]
                        p0 = soft_info.verts_rest[v[3]] + Dm @ b3
                        is_valid, d, grad_mat = func_soft_sdf(e_b, p0, soft_info)
                        if not is_valid:
                            continue
                        grad = (Dm @ Ds_inv).transpose() @ grad_mat
                        h = soft_info.entities_penalty_smoothing_half_distance[e_b]
                        if d > 0.0:
                            continue
                        if kind_a == 1 and soft_info.entities_kind[e_a] != SOFT_KIND_SOLID:
                            normal_a = -gu.qd_normalize(grad, EPS)

                        # Stage displacement of the sample relative to the collider point (which moves with the tetrahedron).
                        pos_b_start = qd.Vector.zero(gs.qd_float, 3)
                        for j in qd.static(range(4)):
                            pos_b_start += bary_b[j] * soft_state.verts_pos_stage_start[v[j], i_b]
                        p_rel = pos_b_start - pos_start
                        d_start = d - grad.dot(p_rel)

                        k = qd.sqrt(k_a * soft_info.entities_penalty_coefficient[e_b])
                        falloff = qd.sqrt(falloff_a * soft_info.entities_friction_falloff_vel[e_b])
                        mu = qd.sqrt(mu_a * soft_info.entities_friction[e_b])
                        c_visc = qd.sqrt(c_visc_a * soft_info.entities_viscous_friction[e_b])
                        c_ndamp = qd.sqrt(c_ndamp_a * soft_info.entities_normal_viscous_damping[e_b])
                        max_align = soft_info.entities_max_alignment_normals[e_b]
                        energy, force, dforce, _ = collision_response(
                            d,
                            grad,
                            normal_a,
                            p_rel,
                            d_start,
                            k,
                            h,
                            0.0,
                            mu,
                            falloff,
                            c_visc,
                            c_ndamp,
                            max_align,
                            mochi_state.dt_stage[i_b],
                            EPS,
                            mochi_config,
                        )
                        wf = w * force
                        D = -w * dforce
                        r_a = pos - dyn_state.links.pos[qd.max(i_la, 0), i_b]
                        is_dynamic_a = kind_a == 1 or mochi_info.links.is_dynamic[i_la]

                        if qd.static(assem_obj):
                            qd.atomic_add(mochi_state.obj[i_b], w * energy)
                        if qd.static(assem_res):
                            if kind_a == 0:
                                if is_dynamic_a:
                                    torque = r_a.cross(wf)
                                    for kk in qd.static(range(3)):
                                        qd.atomic_add(mochi_state.links_res[i_la, i_b][kk], -wf[kk])
                                        qd.atomic_add(mochi_state.links_res[i_la, i_b][3 + kk], -torque[kk])
                            else:
                                tri = soft_info.samples_tri[i_sample]
                                bary = soft_info.samples_bary[i_sample]
                                for i in qd.static(range(3)):
                                    func_add_soft_vec(mochi_state.res, tri[i], i_b, -(bary[i]) * wf, soft_info)
                            for j in qd.static(range(4)):
                                func_add_soft_vec(mochi_state.res, v[j], i_b, bary_b[j] * wf, soft_info)
                        if assem_dres:
                            if kind_a == 0:
                                if is_dynamic_a:
                                    S_a = skew(r_a)
                                    DS = D @ S_a
                                    SD = S_a @ D
                                    SDS = S_a @ D @ S_a
                                    for kk in qd.static(range(3)):
                                        for ll in qd.static(range(3)):
                                            qd.atomic_add(mochi_state.H_diag[i_la, i_b][kk, ll], D[kk, ll])
                                            qd.atomic_add(mochi_state.H_diag[i_la, i_b][kk, 3 + ll], -DS[kk, ll])
                                            qd.atomic_add(mochi_state.H_diag[i_la, i_b][3 + kk, ll], SD[kk, ll])
                                            qd.atomic_add(mochi_state.H_diag[i_la, i_b][3 + kk, 3 + ll], -SDS[kk, ll])
                            else:
                                tri = soft_info.samples_tri[i_sample]
                                bary = soft_info.samples_bary[i_sample]
                                for i in qd.static(range(3)):
                                    qd.atomic_add(soft_state.verts_H_diag[tri[i], i_b], (bary[i] * bary[i]) * D)
                            for j in qd.static(range(4)):
                                qd.atomic_add(soft_state.verts_H_diag[v[j], i_b], (bary_b[j] * bary_b[j]) * D)
                        if assem_dres or record:
                            i_h = qd.atomic_add(soft_state.n_sc_hits[i_b], 1)
                            if i_h < max_hits:
                                soft_state.sc_hit_kind_a[i_h, i_b] = kind_a
                                soft_state.sc_hit_sample_a[i_h, i_b] = i_sample
                                soft_state.sc_hit_link_a[i_h, i_b] = i_la if (kind_a == 0 and is_dynamic_a) else -1
                                soft_state.sc_hit_r_a[i_h, i_b] = r_a
                                soft_state.sc_hit_elem_b[i_h, i_b] = i_el_cur
                                soft_state.sc_hit_bary_b[i_h, i_b] = bary_b
                                soft_state.sc_hit_D[i_h, i_b] = func_mat3_to_sym6(D)
                                if qd.static(record):
                                    hit_readback.sc_hit_force[i_h, i_b] = wf
                                    hit_readback.sc_hit_pos[i_h, i_b] = pos
                                    hit_readback.sc_hit_normal[i_h, i_b] = gu.qd_normalize(grad, EPS)
                                    hit_readback.sc_hit_distance[i_h, i_b] = d
                            else:
                                qd.atomic_or(errno[i_b], array_class.ErrorCode.OVERFLOW_MOCHI_CONTACTS)
                        if qd.static(record):
                            if kind_a == 0:
                                qd.atomic_add(dyn_state.links.contact_force[i_la, i_b], wf)
                            else:
                                tri = soft_info.samples_tri[i_sample]
                                bary = soft_info.samples_bary[i_sample]
                                for i in qd.static(range(3)):
                                    qd.atomic_add(soft_state.verts_contact_force[tri[i], i_b], bary[i] * wf)
                            for j in qd.static(range(4)):
                                qd.atomic_add(soft_state.verts_contact_force[v[j], i_b], -bary_b[j] * wf)
                i_node += 1


@qd.kernel
def kernel_soft_collider_eval(
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    mochi_info: MochiInfo,
    mochi_state: MochiState,
    soft_info: MochiSoftInfo,
    soft_state: MochiSoftState,
    hit_readback: MochiHitReadback,
    rigid_config: qd.template(),
    mochi_config: qd.template(),
    assem_obj: qd.template(),
    assem_res: qd.template(),
    assem_dres: qd.i32,
    skip_ls_done: qd.i32,
    record: qd.template(),
    errno: qd.Tensor,
):
    func_soft_collider_eval(
        0,
        False,
        mochi_state.all_envs,
        mochi_state.n_envs_all,
        dyn_state,
        dyn_info,
        mochi_info,
        mochi_state,
        soft_info,
        soft_state,
        hit_readback,
        rigid_config,
        mochi_config,
        assem_obj,
        assem_res,
        assem_dres,
        skip_ls_done,
        record,
        errno,
    )


@qd.kernel
def kernel_tet_tree_refit(
    mochi_state: MochiState,
    soft_info: MochiSoftInfo,
    soft_state: MochiSoftState,
    rigid_config: qd.template(),
    mochi_config: qd.template(),
    skip_ls_done: qd.i32,
):
    func_tet_tree_refit(
        0,
        False,
        mochi_state.all_envs,
        mochi_state.n_envs_all,
        mochi_state,
        soft_info,
        soft_state,
        rigid_config,
        mochi_config,
        skip_ls_done,
    )


@qd.kernel
def kernel_soft_set_pair_enabled(
    entities_pair_enabled: qd.types.ndarray(),
    soft_info: MochiSoftInfo,
    rigid_config: qd.template(),
):
    n_entities = entities_pair_enabled.shape[0]
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_e, j_e in qd.ndrange(n_entities, n_entities):
        soft_info.entities_pair_enabled[i_e, j_e] = entities_pair_enabled[i_e, j_e]


# ------------------------------------------------------------------------------------
# --------------------------------------- shells -------------------------------------
# ------------------------------------------------------------------------------------


@qd.func
def func_shell_nodes(i_t, soft_info: MochiSoftInfo):
    """The six stencil vertices of a shell triangle (-1 for a missing opposite vertex)."""
    v = soft_info.shell_elems_v[i_t]
    hinge = soft_info.shell_elems_hinge[i_t]
    return qd.Vector([v[0], v[1], v[2], hinge[0], hinge[1], hinge[2]], dt=gs.qd_int)


@qd.func
def func_shell_gather(i_t, i_b, field: qd.Tensor, soft_info: MochiSoftInfo):
    """Positions of the six stencil vertices (missing ones take the first vertex, they are extrapolated later)."""
    nodes = func_shell_nodes(i_t, soft_info)
    x = qd.Matrix.zero(gs.qd_float, 6, 3)
    for a in qd.static(range(6)):
        i_v = nodes[a] if nodes[a] >= 0 else nodes[0]
        pos = field[i_v, i_b]
        for k in qd.static(range(3)):
            x[a, k] = pos[k]
    return x


@qd.func
def func_shell_gather_rest(i_t, soft_info: MochiSoftInfo):
    nodes = func_shell_nodes(i_t, soft_info)
    X = qd.Matrix.zero(gs.qd_float, 6, 3)
    for a in qd.static(range(6)):
        i_v = nodes[a] if nodes[a] >= 0 else nodes[0]
        pos = soft_info.verts_rest[i_v]
        for k in qd.static(range(3)):
            X[a, k] = pos[k]
    return X


@qd.func
def func_shell_missing(i_t, soft_info: MochiSoftInfo):
    hinge = soft_info.shell_elems_hinge[i_t]
    return qd.Vector([hinge[0] < 0, hinge[1] < 0, hinge[2] < 0])


@qd.kernel
def kernel_init_shell_fields(
    shell_elems_v: qd.types.ndarray(),
    shell_elems_hinge: qd.types.ndarray(),
    shell_elems_entity_idx: qd.types.ndarray(),
    soft_info: MochiSoftInfo,
    rigid_config: qd.template(),
):
    """Rest data of the shell triangles (area, inverse metric, rest curvature), lumped vertex masses and point-cloud
    collider weights (nodal areas)."""
    n_tris = shell_elems_v.shape[0]
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_t in range(n_tris):
        for k in qd.static(range(3)):
            soft_info.shell_elems_v[i_t][k] = shell_elems_v[i_t, k]
            soft_info.shell_elems_hinge[i_t][k] = shell_elems_hinge[i_t, k]
        soft_info.shell_elems_entity_idx[i_t] = shell_elems_entity_idx[i_t]
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_t in range(n_tris):
        i_e = soft_info.shell_elems_entity_idx[i_t]
        X = func_shell_gather_rest(i_t, soft_info)
        area, A_inv, B = func_shell_rest_data(X, func_shell_missing(i_t, soft_info), SHELL_TINY)
        soft_info.shell_elems_area[i_t] = area
        soft_info.shell_elems_A_inv[i_t] = A_inv
        soft_info.shell_elems_B[i_t] = B
        rho = soft_info.entities_rho[i_e]
        is_collider = soft_info.entities_collider_type[i_e] == COLLIDER_TYPE.POINT_CLOUD
        for k in qd.static(range(3)):
            i_v = soft_info.shell_elems_v[i_t][k]
            qd.atomic_add(soft_info.verts_mass[i_v], rho * area / 3.0)
            if is_collider:
                qd.atomic_add(soft_info.verts_collider_weight[i_v], area / 3.0)


@qd.func
def func_shell_stage_start(
    i_b_env,
    per_env: qd.template(),
    envs: qd.types.ndarray(),
    n_envs: qd.types.ndarray(),
    soft_info: MochiSoftInfo,
    soft_state: MochiSoftState,
    rigid_config: qd.template(),
):
    """Stage-start membrane and bending strains of every shell triangle (stiffness damping)."""
    n_tris = soft_info.shell_elems_v.shape[0]
    _B = soft_state.verts_pos.shape[1]
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_t, i_slot in qd.ndrange(n_tris, n_envs[None]) if qd.static(not per_env) else qd.ndrange(n_tris, 1):
        i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
        x = func_shell_gather(i_t, i_b, soft_state.verts_pos_stage_start, soft_info)
        X = func_shell_gather_rest(i_t, soft_info)
        eps_m, s = func_shell_strains(
            x,
            X,
            func_shell_missing(i_t, soft_info),
            soft_info.shell_elems_A_inv[i_t],
            soft_info.shell_elems_B[i_t],
            SHELL_TINY,
        )
        soft_state.shell_elems_eps_stage_start[i_t, i_b] = eps_m
        soft_state.shell_elems_s_stage_start[i_t, i_b] = s


@qd.kernel
def kernel_shell_stage_start(
    mochi_state: MochiState,
    soft_info: MochiSoftInfo,
    soft_state: MochiSoftState,
    rigid_config: qd.template(),
):
    func_shell_stage_start(
        0,
        False,
        mochi_state.all_envs,
        mochi_state.n_envs_all,
        soft_info,
        soft_state,
        rigid_config,
    )


@qd.func
def func_shell_assemble(
    i_b_env,
    per_env: qd.template(),
    envs: qd.types.ndarray(),
    n_envs: qd.types.ndarray(),
    mochi_info: MochiInfo,
    mochi_state: MochiState,
    soft_info: MochiSoftInfo,
    soft_state: MochiSoftState,
    rigid_config: qd.template(),
    assem_obj: qd.template(),
    assem_res: qd.template(),
    assem_dres,
    skip_ls_done,
):
    """Membrane, bending, inertia (consistent mass), gravity and damping of every shell triangle of the running
    environments: residual into the vertex degrees of freedom, 18x18 tangent block per triangle and its diagonal
    blocks into the vertex preconditioner."""
    n_tris = soft_info.shell_elems_v.shape[0]
    _B = soft_state.verts_pos.shape[1]

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_t, i_slot in qd.ndrange(n_tris, n_envs[None]) if qd.static(not per_env) else qd.ndrange(n_tris, 1):
        i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
        if not func_is_env_active(i_b, mochi_state, skip_ls_done):
            continue
        i_e = soft_info.shell_elems_entity_idx[i_t]
        nodes = func_shell_nodes(i_t, soft_info)
        is_missing = func_shell_missing(i_t, soft_info)
        h = mochi_state.dt_stage[i_b]
        area = soft_info.shell_elems_area[i_t]
        rho = soft_info.entities_rho[i_e]
        factor = soft_info.entities_stiffness_damping[i_e] / h
        scale = 1.0 + factor
        ss_weight = factor / scale
        x = func_shell_gather(i_t, i_b, soft_state.verts_pos, soft_info)
        X = func_shell_gather_rest(i_t, soft_info)
        energy, res, K, _eps_m, _s = func_shell_elastic(
            x,
            X,
            is_missing,
            area,
            soft_info.shell_elems_A_inv[i_t],
            soft_info.shell_elems_B[i_t],
            scale * soft_info.entities_membrane_lambda[i_e],
            scale * soft_info.entities_membrane_mu[i_e],
            scale * soft_info.entities_bending_alpha[i_e],
            scale * soft_info.entities_bending_beta[i_e],
            ss_weight,
            soft_state.shell_elems_eps_stage_start[i_t, i_b],
            soft_state.shell_elems_s_stage_start[i_t, i_b],
            SHELL_TINY,
            True,
            assem_dres,
        )

        # Inertia, mass damping and gravity on the three vertices with the consistent mass rho A [2 1 1; ...] / 12.
        m_unit = rho * area / 12.0
        c_inertia = m_unit / (h * h)
        c_damping = m_unit * soft_info.entities_mass_damping[i_e] / h
        for f in qd.static(range(3)):
            i_f = nodes[f]
            for g in qd.static(range(3)):
                i_g = nodes[g]
                m_fg = 2.0 if qd.static(f == g) else 1.0
                a_g = (
                    soft_state.verts_pos[i_g, i_b]
                    - soft_state.verts_pos_stage_start[i_g, i_b]
                    - h * soft_state.verts_vel_stage_start[i_g, i_b]
                )
                b_g = soft_state.verts_pos[i_g, i_b] - soft_state.verts_pos_stage_start[i_g, i_b]
                for k in qd.static(range(3)):
                    res[3 * f + k] += m_fg * (c_inertia * a_g[k] + c_damping * b_g[k])
                    if assem_dres:
                        K[3 * f + k, 3 * g + k] += m_fg * (c_inertia + c_damping)
                if qd.static(assem_obj):
                    a_f = (
                        soft_state.verts_pos[i_f, i_b]
                        - soft_state.verts_pos_stage_start[i_f, i_b]
                        - h * soft_state.verts_vel_stage_start[i_f, i_b]
                    )
                    b_f = soft_state.verts_pos[i_f, i_b] - soft_state.verts_pos_stage_start[i_f, i_b]
                    energy += 0.5 * m_fg * (c_inertia * a_f.dot(a_g) + c_damping * b_f.dot(b_g))
        if soft_info.entities_has_gravity[i_e]:
            gravity = mochi_info.gravity[i_b]
            for f in qd.static(range(3)):
                for k in qd.static(range(3)):
                    res[3 * f + k] -= rho * area / 3.0 * gravity[k]
                if qd.static(assem_obj):
                    energy -= rho * area / 3.0 * gravity.dot(soft_state.verts_pos[nodes[f], i_b])

        if qd.static(assem_res):
            for a in qd.static(range(6)):
                if nodes[a] >= 0:
                    func_add_soft_vec(
                        mochi_state.res,
                        nodes[a],
                        i_b,
                        qd.Vector([res[3 * a], res[3 * a + 1], res[3 * a + 2]], dt=gs.qd_float),
                        soft_info,
                    )
        if qd.static(assem_obj):
            qd.atomic_add(mochi_state.obj[i_b], energy)
        if assem_dres:
            for a in qd.static(range(6)):
                if nodes[a] >= 0:
                    for r in qd.static(range(3)):
                        row_start = soft_info.csr_start[3 * nodes[a] + r]
                        for c in qd.static(range(6)):
                            if nodes[c] >= 0:
                                pos = row_start + soft_info.shell_csr_block[i_t, 6 * a + c]
                                for cc in qd.static(range(3)):
                                    qd.atomic_add(soft_state.csr_values[pos + cc, i_b], K[3 * a + r, 3 * c + cc])
            for a in qd.static(range(6)):
                if nodes[a] >= 0:
                    block = qd.Matrix.zero(gs.qd_float, 3, 3)
                    for r in qd.static(range(3)):
                        for c in qd.static(range(3)):
                            block[r, c] = K[3 * a + r, 3 * a + c]
                    qd.atomic_add(soft_state.verts_H_diag[nodes[a], i_b], block)


@qd.kernel
def kernel_shell_assemble(
    mochi_info: MochiInfo,
    mochi_state: MochiState,
    soft_info: MochiSoftInfo,
    soft_state: MochiSoftState,
    rigid_config: qd.template(),
    assem_obj: qd.template(),
    assem_res: qd.template(),
    assem_dres: qd.i32,
    skip_ls_done: qd.i32,
):
    func_shell_assemble(
        0,
        False,
        mochi_state.all_envs,
        mochi_state.n_envs_all,
        mochi_info,
        mochi_state,
        soft_info,
        soft_state,
        rigid_config,
        assem_obj,
        assem_res,
        assem_dres,
        skip_ls_done,
    )


# ------------------------------------------------------------------------------------
# ---------------------------------------- rods --------------------------------------
# ------------------------------------------------------------------------------------


@qd.func
def func_rod_twist_dof(i_r, soft_info: MochiSoftInfo):
    return soft_info.twist_dof_start[None] + i_r


@qd.func
def func_rod_stencil_dofs(i_s, soft_info: MochiSoftInfo):
    """Degrees of freedom of a stencil in the order [x0, theta0, x1, theta1, x2]."""
    v = soft_info.rod_stencils_v[i_s]
    e = soft_info.rod_stencils_e[i_s]
    dofs = qd.Vector.zero(gs.qd_int, 11)
    for k in qd.static(range(3)):
        dofs[k] = func_soft_dof(v[0], k, soft_info)
        dofs[4 + k] = func_soft_dof(v[1], k, soft_info)
        dofs[8 + k] = func_soft_dof(v[2], k, soft_info)
    dofs[3] = func_rod_twist_dof(e[0], soft_info)
    dofs[7] = func_rod_twist_dof(e[1], soft_info)
    return dofs


@qd.kernel
def kernel_init_rod_fields(
    rod_elems_v: qd.types.ndarray(),
    rod_elems_entity_idx: qd.types.ndarray(),
    rod_elems_axis_ref: qd.types.ndarray(),
    rod_stencils_v: qd.types.ndarray(),
    rod_stencils_e: qd.types.ndarray(),
    soft_info: MochiSoftInfo,
    soft_state: MochiSoftState,
    rigid_config: qd.template(),
):
    """Rest data of the rod segments and stencils, lumped node masses, and the initial material axes and twists."""
    n_elems = rod_elems_v.shape[0]
    n_stencils = rod_stencils_v.shape[0]
    _B = soft_state.verts_pos.shape[1]
    tiny = soft_info.rod_tiny[None]
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_r in range(n_elems):
        for k in qd.static(range(2)):
            soft_info.rod_elems_v[i_r][k] = rod_elems_v[i_r, k]
        for k in qd.static(range(3)):
            soft_info.rod_elems_axis_ref[i_r][k] = rod_elems_axis_ref[i_r, k]
        i_e = rod_elems_entity_idx[i_r]
        soft_info.rod_elems_entity_idx[i_r] = i_e
        L = (soft_info.verts_rest[rod_elems_v[i_r, 1]] - soft_info.verts_rest[rod_elems_v[i_r, 0]]).norm()
        soft_info.rod_elems_L[i_r] = L
        soft_info.rod_elems_rot_inertia[i_r] = soft_info.entities_rot_inertia[i_e] * L
        half_mass = 0.5 * soft_info.entities_rho[i_e] * L
        qd.atomic_add(soft_info.verts_mass[rod_elems_v[i_r, 0]], half_mass)
        qd.atomic_add(soft_info.verts_mass[rod_elems_v[i_r, 1]], half_mass)
        if soft_info.entities_collider_type[i_e] == COLLIDER_TYPE.POINT_CLOUD:
            qd.atomic_add(soft_info.verts_collider_weight[rod_elems_v[i_r, 0]], 0.5 * L)
            qd.atomic_add(soft_info.verts_collider_weight[rod_elems_v[i_r, 1]], 0.5 * L)
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_s in range(n_stencils):
        for k in qd.static(range(3)):
            soft_info.rod_stencils_v[i_s][k] = rod_stencils_v[i_s, k]
        for k in qd.static(range(2)):
            soft_info.rod_stencils_e[i_s][k] = rod_stencils_e[i_s, k]
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_s in range(n_stencils):
        v = soft_info.rod_stencils_v[i_s]
        e = soft_info.rod_stencils_e[i_s]
        X0 = soft_info.verts_rest[v[0]]
        X1 = soft_info.verts_rest[v[1]]
        X2 = soft_info.verts_rest[v[2]]
        L = 0.5 * ((X1 - X0).norm() + (X2 - X1).norm())
        soft_info.rod_stencils_L[i_s] = L
        ka, kb, tw = func_rod_bend_twist_measures(
            X0, X1, X2, soft_info.rod_elems_axis_ref[e[0]], soft_info.rod_elems_axis_ref[e[1]], L, tiny
        )
        soft_info.rod_stencils_ref[i_s] = qd.Vector([ka, kb, tw], dt=gs.qd_float)
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_r, i_b in qd.ndrange(n_elems, _B):
        axis = soft_info.rod_elems_axis_ref[i_r]
        soft_state.rod_elems_axis[i_r, i_b] = axis
        soft_state.rod_elems_axis_stage_start[i_r, i_b] = axis
        soft_state.rod_elems_axis_ls_ref[i_r, i_b] = axis
        soft_state.rod_elems_twist[i_r, i_b] = 0.0
        soft_state.rod_elems_twist_vel[i_r, i_b] = 0.0
        for k in qd.static(range(N_HISTORY)):
            soft_state.rod_elems_twist_prev[k, i_r, i_b] = 0.0
            soft_state.rod_elems_twist_vel_prev[k, i_r, i_b] = 0.0
        soft_state.rod_elems_twist_step_start[i_r, i_b] = 0.0
        soft_state.rod_elems_twist_vel_stage_start[i_r, i_b] = 0.0
        soft_state.rod_elems_twist_ls_ref[i_r, i_b] = 0.0


@qd.func
def func_rod_tangent(i_r, i_b, field: qd.Tensor, soft_info: MochiSoftInfo):
    v = soft_info.rod_elems_v[i_r]
    return func_rod_normalize(field[v[1], i_b] - field[v[0], i_b], soft_info.rod_tiny[None])


@qd.func
def func_rod_step_start(
    i_b_env,
    per_env: qd.template(),
    envs: qd.types.ndarray(),
    n_envs: qd.types.ndarray(),
    mochi_state: MochiState,
    soft_info: MochiSoftInfo,
    soft_state: MochiSoftState,
    rigid_config: qd.template(),
    mochi_config: qd.template(),
):
    """Recenter the twist of every segment to zero (shifting its history), build the stage-start twist, twist rate and
    material axes (transported from the previous positions to the stage-start ones), and the stage-start strain
    measures. Runs after the vertex step start."""
    n_elems = soft_state.rod_elems_H.shape[0]
    n_stencils = soft_state.rod_stencils_H.shape[0]
    _B = soft_state.verts_pos.shape[1]
    tiny = soft_info.rod_tiny[None]
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_r, i_slot in qd.ndrange(n_elems, n_envs[None]) if qd.static(not per_env) else qd.ndrange(n_elems, 1):
        i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
        theta = soft_state.rod_elems_twist[i_r, i_b]
        soft_state.rod_elems_twist_prev[1, i_r, i_b] = soft_state.rod_elems_twist_prev[0, i_r, i_b] - theta
        soft_state.rod_elems_twist_prev[0, i_r, i_b] = 0.0
        soft_state.rod_elems_twist_vel_prev[1, i_r, i_b] = soft_state.rod_elems_twist_vel_prev[0, i_r, i_b]
        soft_state.rod_elems_twist_vel_prev[0, i_r, i_b] = soft_state.rod_elems_twist_vel[i_r, i_b]
        twist_start = gs.qd_float(0.0)
        vel_start = soft_state.rod_elems_twist_vel[i_r, i_b]
        if qd.static(mochi_config.integrator == INTEGRATOR.BDF2):  # noqa: SIM102
            if mochi_state.n_hist[i_b] >= 2:
                twist_start = BDF2_ALPHA_2 * soft_state.rod_elems_twist_prev[1, i_r, i_b]
                vel_start = vel_start + BDF2_ALPHA_2 * (soft_state.rod_elems_twist_vel_prev[1, i_r, i_b] - vel_start)
        soft_state.rod_elems_twist_step_start[i_r, i_b] = twist_start
        soft_state.rod_elems_twist[i_r, i_b] = twist_start
        soft_state.rod_elems_twist_vel_stage_start[i_r, i_b] = vel_start
        # Axes follow the tangents from the previous positions to the stage-start ones, plus the start twist.
        v_prev = soft_info.rod_elems_v[i_r]
        t_prev = func_rod_normalize(
            soft_state.verts_pos_prev[0, v_prev[1], i_b] - soft_state.verts_pos_prev[0, v_prev[0], i_b], tiny
        )
        t_start = func_rod_tangent(i_r, i_b, soft_state.verts_pos_stage_start, soft_info)
        axis = func_rod_transport_axis(t_prev, t_start, twist_start, soft_state.rod_elems_axis[i_r, i_b], tiny)
        soft_state.rod_elems_axis_stage_start[i_r, i_b] = axis
        # The warm start differs from the stage start where fixed nodes jumped to their prescribed positions.
        t_warm = func_rod_tangent(i_r, i_b, soft_state.verts_pos, soft_info)
        soft_state.rod_elems_axis[i_r, i_b] = func_rod_transport_axis(t_start, t_warm, 0.0, axis, tiny)
        v = soft_info.rod_elems_v[i_r]
        strain, _q = func_rod_axial_strain(
            soft_state.verts_pos_stage_start[v[0], i_b],
            soft_state.verts_pos_stage_start[v[1], i_b],
            qd.max(soft_info.rod_elems_L[i_r], tiny),
        )
        soft_state.rod_elems_strain_stage_start[i_r, i_b] = strain
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_s, i_slot in qd.ndrange(n_stencils, n_envs[None]) if qd.static(not per_env) else qd.ndrange(n_stencils, 1):
        i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
        v = soft_info.rod_stencils_v[i_s]
        e = soft_info.rod_stencils_e[i_s]
        ka, kb, tw = func_rod_bend_twist_measures(
            soft_state.verts_pos_stage_start[v[0], i_b],
            soft_state.verts_pos_stage_start[v[1], i_b],
            soft_state.verts_pos_stage_start[v[2], i_b],
            soft_state.rod_elems_axis_stage_start[e[0], i_b],
            soft_state.rod_elems_axis_stage_start[e[1], i_b],
            qd.max(soft_info.rod_stencils_L[i_s], tiny),
            tiny,
        )
        soft_state.rod_stencils_stage_start[i_s, i_b] = qd.Vector([ka, kb, tw], dt=gs.qd_float)


@qd.kernel
def kernel_rod_step_start(
    mochi_state: MochiState,
    soft_info: MochiSoftInfo,
    soft_state: MochiSoftState,
    rigid_config: qd.template(),
    mochi_config: qd.template(),
):
    func_rod_step_start(
        0,
        False,
        mochi_state.all_envs,
        mochi_state.n_envs_all,
        mochi_state,
        soft_info,
        soft_state,
        rigid_config,
        mochi_config,
    )


@qd.func
def func_rod_assemble(
    i_b_env,
    per_env: qd.template(),
    envs: qd.types.ndarray(),
    n_envs: qd.types.ndarray(),
    mochi_info: MochiInfo,
    mochi_state: MochiState,
    soft_info: MochiSoftInfo,
    soft_state: MochiSoftState,
    rigid_config: qd.template(),
    assem_obj: qd.template(),
    assem_res: qd.template(),
    assem_dres,
    skip_ls_done,
):
    """Stretching, lumped inertia, gravity and damping of every rod segment (with the twist inertia on its twist
    degree of freedom), and bending and twisting of every interior stencil."""
    n_elems = soft_state.rod_elems_H.shape[0]
    n_stencils = soft_state.rod_stencils_H.shape[0]
    _B = soft_state.verts_pos.shape[1]
    EPS = mochi_info.EPS[None]

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_r, i_slot in qd.ndrange(n_elems, n_envs[None]) if qd.static(not per_env) else qd.ndrange(n_elems, 1):
        i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
        if not func_is_env_active(i_b, mochi_state, skip_ls_done):
            continue
        L = soft_info.rod_elems_L[i_r]
        if L <= 0.0:
            continue
        i_e = soft_info.rod_elems_entity_idx[i_r]
        v = soft_info.rod_elems_v[i_r]
        h = mochi_state.dt_stage[i_b]
        f = soft_info.entities_stiffness_damping[i_e] / h
        x0 = soft_state.verts_pos[v[0], i_b]
        x1 = soft_state.verts_pos[v[1], i_b]
        energy, g, T = func_rod_axial(
            x0,
            x1,
            L,
            soft_info.entities_axial_stiffness[i_e],
            f,
            soft_state.rod_elems_strain_stage_start[i_r, i_b],
            EPS,
            assem_dres,
        )
        # Lumped inertia, mass damping and gravity of the two nodes (half the segment mass each) and twist inertia.
        half_mass = 0.5 * soft_info.entities_rho[i_e] * L
        c_inertia = half_mass / (h * h)
        c_damping = half_mass * soft_info.entities_mass_damping[i_e] / h
        gravity = mochi_info.gravity[i_b] * (1.0 if soft_info.entities_has_gravity[i_e] else 0.0)
        res0 = -g
        res1 = g
        for k in qd.static(range(2)):
            i_v = v[k]
            a = (
                soft_state.verts_pos[i_v, i_b]
                - soft_state.verts_pos_stage_start[i_v, i_b]
                - h * soft_state.verts_vel_stage_start[i_v, i_b]
            )
            b = soft_state.verts_pos[i_v, i_b] - soft_state.verts_pos_stage_start[i_v, i_b]
            node_res = c_inertia * a + c_damping * b - half_mass * gravity
            if qd.static(k == 0):
                res0 += node_res
            else:
                res1 += node_res
            if qd.static(assem_obj):
                energy += 0.5 * (c_inertia * a.norm_sqr() + c_damping * b.norm_sqr()) - half_mass * gravity.dot(
                    soft_state.verts_pos[i_v, i_b]
                )
        I_rot = soft_info.rod_elems_rot_inertia[i_r]
        c_rot = I_rot / (h * h)
        c_rot_damping = I_rot * soft_info.entities_mass_damping[i_e] / h
        a_tw = (
            soft_state.rod_elems_twist[i_r, i_b]
            - soft_state.rod_elems_twist_step_start[i_r, i_b]
            - h * soft_state.rod_elems_twist_vel_stage_start[i_r, i_b]
        )
        b_tw = soft_state.rod_elems_twist[i_r, i_b] - soft_state.rod_elems_twist_step_start[i_r, i_b]
        i_d = func_rod_twist_dof(i_r, soft_info)
        if qd.static(assem_res):
            func_add_soft_vec(mochi_state.res, v[0], i_b, res0, soft_info)
            func_add_soft_vec(mochi_state.res, v[1], i_b, res1, soft_info)
            qd.atomic_add(mochi_state.res[i_d, i_b], c_rot * a_tw + c_rot_damping * b_tw)
        if qd.static(assem_obj):
            energy += 0.5 * (c_rot * a_tw * a_tw + c_rot_damping * b_tw * b_tw)
            qd.atomic_add(mochi_state.obj[i_b], energy)
        if assem_dres:
            soft_state.rod_elems_H[i_r, i_b] = T
            block = T + (c_inertia + c_damping) * qd.Matrix.identity(gs.qd_float, 3)
            qd.atomic_add(soft_state.verts_H_diag[v[0], i_b], block)
            qd.atomic_add(soft_state.verts_H_diag[v[1], i_b], block)
            qd.atomic_add(mochi_state.dofs_H_diag[i_d, i_b], c_rot + c_rot_damping)
            # The nodal inertia is not part of the segment block: it is applied through the vertex lumped term.
            soft_state.rod_elems_inertia[i_r, i_b] = c_inertia + c_damping
            for r in qd.static(range(3)):
                for c in qd.static(range(3)):
                    diagonal = block[r, c] - T[r, c]
                    qd.atomic_add(
                        soft_state.csr_values[soft_info.rod_elems_csr[i_r, 6 * r + c], i_b], T[r, c] + diagonal
                    )
                    qd.atomic_add(
                        soft_state.csr_values[soft_info.rod_elems_csr[i_r, 6 * (3 + r) + 3 + c], i_b],
                        T[r, c] + diagonal,
                    )
                    qd.atomic_add(soft_state.csr_values[soft_info.rod_elems_csr[i_r, 6 * r + 3 + c], i_b], -T[r, c])
                    qd.atomic_add(soft_state.csr_values[soft_info.rod_elems_csr[i_r, 6 * (3 + r) + c], i_b], -T[r, c])

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_s, i_slot in qd.ndrange(n_stencils, n_envs[None]) if qd.static(not per_env) else qd.ndrange(n_stencils, 1):
        i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
        if not func_is_env_active(i_b, mochi_state, skip_ls_done):
            continue
        L = soft_info.rod_stencils_L[i_s]
        if L <= 0.0:
            continue
        v = soft_info.rod_stencils_v[i_s]
        e = soft_info.rod_stencils_e[i_s]
        i_e = soft_info.rod_elems_entity_idx[e[0]]
        h = mochi_state.dt_stage[i_b]
        f = soft_info.entities_stiffness_damping[i_e] / h
        ref = soft_info.rod_stencils_ref[i_s]
        ss = soft_state.rod_stencils_stage_start[i_s, i_b]
        energy, res, K, _ka, _kb, _tw = func_rod_bend_twist(
            soft_state.verts_pos[v[0], i_b],
            soft_state.verts_pos[v[1], i_b],
            soft_state.verts_pos[v[2], i_b],
            soft_state.rod_elems_axis[e[0], i_b],
            soft_state.rod_elems_axis[e[1], i_b],
            L,
            ref[0],
            ref[1],
            ref[2],
            ss[0],
            ss[1],
            ss[2],
            soft_info.entities_membrane_mu[i_e],
            soft_info.entities_membrane_lambda[i_e],
            soft_info.entities_torsional_stiffness[i_e],
            f,
            soft_info.rod_tiny[None],
            assem_dres,
        )
        dofs = func_rod_stencil_dofs(i_s, soft_info)
        if qd.static(assem_res):
            for p in qd.static(range(11)):
                qd.atomic_add(mochi_state.res[dofs[p], i_b], res[p])
        if qd.static(assem_obj):
            qd.atomic_add(mochi_state.obj[i_b], energy)
        if assem_dres:
            soft_state.rod_stencils_H[i_s, i_b] = K
            for p in qd.static(range(11)):
                for q in qd.static(range(11)):
                    qd.atomic_add(soft_state.csr_values[soft_info.rod_stencils_csr[i_s, 11 * p + q], i_b], K[p, q])
            for a in qd.static(range(3)):
                block = qd.Matrix.zero(gs.qd_float, 3, 3)
                for r in qd.static(range(3)):
                    for c in qd.static(range(3)):
                        block[r, c] = K[4 * a + r, 4 * a + c]
                qd.atomic_add(soft_state.verts_H_diag[v[a], i_b], block)
            qd.atomic_add(soft_state.rod_elems_twist_pcg[e[0], i_b], K[3, 3])
            qd.atomic_add(soft_state.rod_elems_twist_pcg[e[1], i_b], K[7, 7])


@qd.kernel
def kernel_rod_assemble(
    mochi_info: MochiInfo,
    mochi_state: MochiState,
    soft_info: MochiSoftInfo,
    soft_state: MochiSoftState,
    rigid_config: qd.template(),
    assem_obj: qd.template(),
    assem_res: qd.template(),
    assem_dres: qd.i32,
    skip_ls_done: qd.i32,
):
    func_rod_assemble(
        0,
        False,
        mochi_state.all_envs,
        mochi_state.n_envs_all,
        mochi_info,
        mochi_state,
        soft_info,
        soft_state,
        rigid_config,
        assem_obj,
        assem_res,
        assem_dres,
        skip_ls_done,
    )


@qd.func
def func_rod_apply_increment(
    i_b_env,
    per_env: qd.template(),
    envs: qd.types.ndarray(),
    n_envs: qd.types.ndarray(),
    mochi_state: MochiState,
    soft_info: MochiSoftInfo,
    soft_state: MochiSoftState,
    rigid_config: qd.template(),
):
    """Trial twist of every segment and its material axis: parallel transport from the reference iterate's tangent
    to the trial tangent (the vertex increment has been applied), then rotation by the twist increment."""
    n_elems = soft_state.rod_elems_H.shape[0]
    _B = soft_state.verts_pos.shape[1]
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_r, i_slot in qd.ndrange(n_elems, n_envs[None]) if qd.static(not per_env) else qd.ndrange(n_elems, 1):
        i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
        if not func_is_env_active(i_b, mochi_state, True) or soft_info.rod_elems_L[i_r] <= 0.0:
            continue
        delta = -mochi_state.ls_alpha[i_b] * mochi_state.dx[func_rod_twist_dof(i_r, soft_info), i_b]
        soft_state.rod_elems_twist[i_r, i_b] = soft_state.rod_elems_twist_ls_ref[i_r, i_b] + delta
        t_ref = func_rod_tangent(i_r, i_b, soft_state.verts_pos_ls_ref, soft_info)
        t_new = func_rod_tangent(i_r, i_b, soft_state.verts_pos, soft_info)
        soft_state.rod_elems_axis[i_r, i_b] = func_rod_transport_axis(
            t_ref, t_new, delta, soft_state.rod_elems_axis_ls_ref[i_r, i_b], soft_info.rod_tiny[None]
        )


@qd.kernel
def kernel_rod_apply_increment(
    mochi_state: MochiState,
    soft_info: MochiSoftInfo,
    soft_state: MochiSoftState,
    rigid_config: qd.template(),
):
    func_rod_apply_increment(
        0,
        False,
        mochi_state.all_envs,
        mochi_state.n_envs_all,
        mochi_state,
        soft_info,
        soft_state,
        rigid_config,
    )


@qd.func
def func_rod_store_ls_ref(
    i_b_env,
    per_env: qd.template(),
    envs: qd.types.ndarray(),
    n_envs: qd.types.ndarray(),
    mochi_state: MochiState,
    soft_state: MochiSoftState,
    rigid_config: qd.template(),
    only_done,
):
    n_elems = soft_state.rod_elems_H.shape[0]
    _B = soft_state.verts_pos.shape[1]
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_r, i_slot in qd.ndrange(n_elems, n_envs[None]) if qd.static(not per_env) else qd.ndrange(n_elems, 1):
        i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
        is_ref = mochi_state.is_active[i_b]
        if only_done:
            is_ref = is_ref and mochi_state.ls_is_done[i_b]
        if is_ref:
            soft_state.rod_elems_twist_ls_ref[i_r, i_b] = soft_state.rod_elems_twist[i_r, i_b]
            soft_state.rod_elems_axis_ls_ref[i_r, i_b] = soft_state.rod_elems_axis[i_r, i_b]


@qd.kernel
def kernel_rod_store_ls_ref(
    mochi_state: MochiState,
    soft_state: MochiSoftState,
    rigid_config: qd.template(),
    only_done: qd.template(),
):
    func_rod_store_ls_ref(
        0,
        False,
        mochi_state.all_envs,
        mochi_state.n_envs_all,
        mochi_state,
        soft_state,
        rigid_config,
        only_done,
    )


@qd.func
def func_rod_post_stage(
    i_b_env,
    per_env: qd.template(),
    envs: qd.types.ndarray(),
    n_envs: qd.types.ndarray(),
    mochi_state: MochiState,
    soft_info: MochiSoftInfo,
    soft_state: MochiSoftState,
    rigid_config: qd.template(),
):
    """Finite-difference twist rate of every segment; a diverged environment falls back to the rest frame at rest.
    The material axes must be restored at the same time level as the vertices (kernel_soft_post_stage puts them back
    at rest), otherwise they stop being orthogonal to the tangents they are transported along."""
    n_elems = soft_state.rod_elems_H.shape[0]
    _B = soft_state.verts_pos.shape[1]
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_r, i_slot in qd.ndrange(n_elems, n_envs[None]) if qd.static(not per_env) else qd.ndrange(n_elems, 1):
        i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
        if mochi_state.status[i_b] == SOLVE_STATUS.DIVERGED:
            soft_state.rod_elems_twist[i_r, i_b] = 0.0
            soft_state.rod_elems_axis[i_r, i_b] = soft_info.rod_elems_axis_ref[i_r]
            soft_state.rod_elems_twist_vel[i_r, i_b] = 0.0
            continue
        h = mochi_state.dt_stage[i_b]
        soft_state.rod_elems_twist_vel[i_r, i_b] = (
            soft_state.rod_elems_twist[i_r, i_b] - soft_state.rod_elems_twist_step_start[i_r, i_b]
        ) / h


@qd.kernel
def kernel_rod_post_stage(
    mochi_state: MochiState,
    soft_info: MochiSoftInfo,
    soft_state: MochiSoftState,
    rigid_config: qd.template(),
):
    func_rod_post_stage(
        0,
        False,
        mochi_state.all_envs,
        mochi_state.n_envs_all,
        mochi_state,
        soft_info,
        soft_state,
        rigid_config,
    )


@qd.func
def func_rod_update_conv_weights(
    i_b_env,
    per_env: qd.template(),
    envs: qd.types.ndarray(),
    n_envs: qd.types.ndarray(),
    mochi_info: MochiInfo,
    mochi_state: MochiState,
    soft_info: MochiSoftInfo,
    soft_state: MochiSoftState,
    rigid_config: qd.template(),
):
    """Convergence weights of the twist degrees of freedom from the rotational inertia: 1 / ((a_ref / r_gyr)^2 sum_e
    I_e I_e) with the gyration radius sqrt(I_lin / rho_lin) of the entity."""
    n_elems = soft_state.rod_elems_H.shape[0]
    _B = soft_state.verts_pos.shape[1]
    EPS = mochi_info.EPS[None]
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_r, i_slot in qd.ndrange(n_elems, n_envs[None]) if qd.static(not per_env) else qd.ndrange(n_elems, 1):
        i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
        i_e = soft_info.rod_elems_entity_idx[i_r]
        w = gs.qd_float(1.0)
        I_e = soft_info.rod_elems_rot_inertia[i_r]
        if I_e > 0.0 and soft_info.entities_rot_inertia[i_e] > 0.0:
            a_ref = qd.max(1.0, mochi_info.gravity[i_b].norm())
            gyr_sq = soft_info.entities_rot_inertia[i_e] / qd.max(soft_info.entities_rho[i_e], EPS)
            total = gs.qd_float(0.0)
            for j_r in range(n_elems):
                if soft_info.rod_elems_entity_idx[j_r] == i_e:
                    total += soft_info.rod_elems_rot_inertia[j_r]
            w = gyr_sq / (a_ref * a_ref * qd.max(total, EPS) * I_e)
        mochi_state.conv_w[func_rod_twist_dof(i_r, soft_info), i_b] = w


@qd.kernel
def kernel_rod_update_conv_weights(
    mochi_info: MochiInfo,
    mochi_state: MochiState,
    soft_info: MochiSoftInfo,
    soft_state: MochiSoftState,
    rigid_config: qd.template(),
):
    func_rod_update_conv_weights(
        0,
        False,
        mochi_state.all_envs,
        mochi_state.n_envs_all,
        mochi_info,
        mochi_state,
        soft_info,
        soft_state,
        rigid_config,
    )


# ------------------------------------------------------------------------------------
# ------------------------------ point-cloud colliders (shells) ----------------------
# ------------------------------------------------------------------------------------


@qd.func
def func_pc_collider_eval(
    i_b_env,
    per_env: qd.template(),
    envs: qd.types.ndarray(),
    n_envs: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    mochi_info: MochiInfo,
    mochi_state: MochiState,
    soft_info: MochiSoftInfo,
    soft_state: MochiSoftState,
    hit_readback: MochiHitReadback,
    rigid_config: qd.template(),
    mochi_config: qd.template(),
    assem_obj: qd.template(),
    assem_res: qd.template(),
    assem_dres,
    skip_ls_done,
    record: qd.template(),
    errno: qd.Tensor,
):
    """Evaluate the sample points of the point-cloud entities against the collider spheres of the vertices found in
    the 27 hash cells around them: signed distance |p - x_b| - r with radial gradient, contact stiffness scaled by the
    nodal area over r^2, response on the sample and on the vertex. Rigid samples never query the spheres (mochi's
    rule: point-cloud colliders only collide with each other)."""
    n_bins = soft_state.pc_hash_heads.shape[0]
    n_verts = soft_state.verts_pos.shape[0]
    _B = soft_state.verts_pos.shape[1]
    max_hits = soft_state.pc_hit_kind_a.shape[0]
    EPS = mochi_info.EPS[None]
    inv_cell = 1.0 / soft_info.pc_hash_cell[None]
    n_rigid = soft_info.n_rigid_queries[None]
    n_soft_queries = soft_info.n_queries[None] - n_rigid

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_q_soft, i_slot in (
        qd.ndrange(n_soft_queries, n_envs[None]) if qd.static(not per_env) else qd.ndrange(n_soft_queries, 1)
    ):
        i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
        if not func_is_env_active(i_b, mochi_state, skip_ls_done):
            continue
        # A deformable sample whose entity has no sphere collider to hit never queries.
        if soft_info.entities_queries_spheres[soft_info.samples_entity_idx[i_q_soft]] == 0:
            continue
        i_q = n_rigid + i_q_soft
        kind_a, i_la, e_a, i_sample, pos, pos_start, normal_a0, w, k_a, falloff_a, mu_a, c_visc_a, c_ndamp_a = (
            func_query_point(i_q, i_b, dyn_state, mochi_info, mochi_state, soft_info, soft_state, EPS)
        )
        is_shell_a = kind_a == 1 and soft_info.entities_kind[e_a] != SOFT_KIND_SOLID
        cell_q = func_hash_cell(pos, inv_cell)
        entry = soft_state.pc_hash_heads[func_hash_bin(cell_q, n_bins - 1), i_b]
        for _walk in range(8 * n_verts):
            if entry < 0:
                break
            i_vb = entry // 8
            k_cell = entry % 8
            entry = soft_state.pc_hash_next[entry, i_b]
            e_b = soft_info.verts_entity_idx[i_vb]
            is_enabled = True
            if kind_a == 0:
                is_enabled = soft_info.entities_links_pair_enabled[e_b, i_la]
            elif e_a == e_b:
                is_enabled = soft_info.entities_self_contact[e_a] != 0
            else:
                is_enabled = soft_info.entities_pair_enabled[e_a, e_b]
            if not is_enabled:
                continue
            x_b = soft_state.verts_pos[i_vb, i_b]
            # Two cells hashed to the same bin share a chain: keep the entries of this cell only (only collider
            # vertices were inserted).
            pad_b = soft_info.entities_collider_radius[e_b] + soft_info.entities_penalty_threshold[e_b]
            if (func_hash_cell(x_b - pad_b, inv_cell) + func_cell_offset(k_cell) != cell_q).any():
                continue
            normal_a = normal_a0
            w_b = soft_info.verts_collider_weight[i_vb]

            diff = pos - x_b
            dist = diff.norm()
            if dist <= EPS:
                continue
            grad = diff / dist
            radius = soft_info.entities_collider_radius[e_b]
            d = dist - radius
            thr = soft_info.entities_penalty_threshold[e_b]
            h = soft_info.entities_penalty_smoothing_half_distance[e_b]
            # Beyond the penalty threshold the penalty and its derivatives vanish: the pair is not a contact
            # (mochi's contact range is the radius plus the threshold).
            if d > thr:
                continue
            if kind_a == 1 and e_a == e_b:
                # Self-contact: samples lying near the vertex in the rest configuration (its own and the neighboring
                # elements) never collide with the sphere of that vertex.
                tri_a = soft_info.samples_tri[i_sample]
                bary_a = soft_info.samples_bary[i_sample]
                rest_a = (
                    bary_a[0] * soft_info.verts_rest[tri_a[0]]
                    + bary_a[1] * soft_info.verts_rest[tri_a[1]]
                    + bary_a[2] * soft_info.verts_rest[tri_a[2]]
                )
                exclusion = radius * soft_info.entities_self_contact_exclusion_ratio[e_b] + thr
                if (rest_a - soft_info.verts_rest[i_vb]).norm() < exclusion:
                    continue
            if is_shell_a:
                normal_a = -grad
            p_rel = (pos - pos_start) - (x_b - soft_state.verts_pos_stage_start[i_vb, i_b])
            d_start = d - grad.dot(p_rel)

            # Dimensional correction of the point-cloud measure: radius^-2 for a surface (shell), radius^-1 for a curve (rod).
            length_scale = radius * radius
            if soft_info.entities_kind[e_b] == SOFT_KIND_ROD:
                length_scale = radius
            k = qd.sqrt(k_a * soft_info.entities_penalty_coefficient[e_b]) * w_b / length_scale
            falloff = qd.sqrt(falloff_a * soft_info.entities_friction_falloff_vel[e_b])
            mu = qd.sqrt(mu_a * soft_info.entities_friction[e_b])
            c_visc = qd.sqrt(c_visc_a * soft_info.entities_viscous_friction[e_b])
            c_ndamp = qd.sqrt(c_ndamp_a * soft_info.entities_normal_viscous_damping[e_b])
            max_align = soft_info.entities_max_alignment_normals[e_b]
            energy, force, dforce, _ = collision_response(
                d,
                grad,
                normal_a,
                p_rel,
                d_start,
                k,
                h,
                thr,
                mu,
                falloff,
                c_visc,
                c_ndamp,
                max_align,
                mochi_state.dt_stage[i_b],
                EPS,
                mochi_config,
            )
            wf = w * force
            D = -w * dforce
            r_a = pos - dyn_state.links.pos[qd.max(i_la, 0), i_b]
            is_dynamic_a = kind_a == 1 or mochi_info.links.is_dynamic[i_la]

            if qd.static(assem_obj):
                qd.atomic_add(mochi_state.obj[i_b], w * energy)
            if qd.static(assem_res):
                if kind_a == 0:
                    if is_dynamic_a:
                        torque = r_a.cross(wf)
                        for kk in qd.static(range(3)):
                            qd.atomic_add(mochi_state.links_res[i_la, i_b][kk], -wf[kk])
                            qd.atomic_add(mochi_state.links_res[i_la, i_b][3 + kk], -torque[kk])
                else:
                    tri = soft_info.samples_tri[i_sample]
                    bary = soft_info.samples_bary[i_sample]
                    for i in qd.static(range(3)):
                        func_add_soft_vec(mochi_state.res, tri[i], i_b, -(bary[i]) * wf, soft_info)
                func_add_soft_vec(mochi_state.res, i_vb, i_b, wf, soft_info)
            if assem_dres:
                if kind_a == 0:
                    if is_dynamic_a:
                        S_a = skew(r_a)
                        DS = D @ S_a
                        SD = S_a @ D
                        SDS = S_a @ D @ S_a
                        for kk in qd.static(range(3)):
                            for ll in qd.static(range(3)):
                                qd.atomic_add(mochi_state.H_diag[i_la, i_b][kk, ll], D[kk, ll])
                                qd.atomic_add(mochi_state.H_diag[i_la, i_b][kk, 3 + ll], -DS[kk, ll])
                                qd.atomic_add(mochi_state.H_diag[i_la, i_b][3 + kk, ll], SD[kk, ll])
                                qd.atomic_add(mochi_state.H_diag[i_la, i_b][3 + kk, 3 + ll], -SDS[kk, ll])
                else:
                    tri = soft_info.samples_tri[i_sample]
                    bary = soft_info.samples_bary[i_sample]
                    for i in qd.static(range(3)):
                        qd.atomic_add(soft_state.verts_H_diag[tri[i], i_b], (bary[i] * bary[i]) * D)
                qd.atomic_add(soft_state.verts_H_diag[i_vb, i_b], D)
            if assem_dres or record:
                i_h = qd.atomic_add(soft_state.n_pc_hits[i_b], 1)
                if i_h < max_hits:
                    soft_state.pc_hit_kind_a[i_h, i_b] = kind_a
                    soft_state.pc_hit_sample_a[i_h, i_b] = i_sample
                    soft_state.pc_hit_link_a[i_h, i_b] = i_la if (kind_a == 0 and is_dynamic_a) else -1
                    soft_state.pc_hit_r_a[i_h, i_b] = r_a
                    soft_state.pc_hit_vert_b[i_h, i_b] = i_vb
                    soft_state.pc_hit_D[i_h, i_b] = func_mat3_to_sym6(D)
                    if qd.static(record):
                        hit_readback.pc_hit_force[i_h, i_b] = wf
                        hit_readback.pc_hit_pos[i_h, i_b] = pos
                        hit_readback.pc_hit_normal[i_h, i_b] = grad
                        hit_readback.pc_hit_distance[i_h, i_b] = d
                else:
                    qd.atomic_or(errno[i_b], array_class.ErrorCode.OVERFLOW_MOCHI_CONTACTS)
            if qd.static(record):
                if kind_a == 0:
                    qd.atomic_add(dyn_state.links.contact_force[i_la, i_b], wf)
                else:
                    tri = soft_info.samples_tri[i_sample]
                    bary = soft_info.samples_bary[i_sample]
                    for i in qd.static(range(3)):
                        qd.atomic_add(soft_state.verts_contact_force[tri[i], i_b], bary[i] * wf)
                qd.atomic_add(soft_state.verts_contact_force[i_vb, i_b], -wf)


@qd.kernel
def kernel_pc_collider_eval(
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    mochi_info: MochiInfo,
    mochi_state: MochiState,
    soft_info: MochiSoftInfo,
    soft_state: MochiSoftState,
    hit_readback: MochiHitReadback,
    rigid_config: qd.template(),
    mochi_config: qd.template(),
    assem_obj: qd.template(),
    assem_res: qd.template(),
    assem_dres: qd.i32,
    skip_ls_done: qd.i32,
    record: qd.template(),
    errno: qd.Tensor,
):
    func_pc_collider_eval(
        0,
        False,
        mochi_state.all_envs,
        mochi_state.n_envs_all,
        dyn_state,
        dyn_info,
        mochi_info,
        mochi_state,
        soft_info,
        soft_state,
        hit_readback,
        rigid_config,
        mochi_config,
        assem_obj,
        assem_res,
        assem_dres,
        skip_ls_done,
        record,
        errno,
    )


@qd.kernel
def kernel_pc_hash_build(
    mochi_state: MochiState,
    soft_info: MochiSoftInfo,
    soft_state: MochiSoftState,
    rigid_config: qd.template(),
    skip_ls_done: qd.i32,
):
    func_pc_hash_build(
        0,
        False,
        mochi_state.all_envs,
        mochi_state.n_envs_all,
        mochi_state,
        soft_info,
        soft_state,
        rigid_config,
        skip_ls_done,
    )
