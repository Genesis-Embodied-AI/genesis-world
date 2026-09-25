"""
Utility functions for IPC coupler.

Stateless helper functions extracted from IPCCoupler for clarity.
"""

from typing import TYPE_CHECKING

import numba as nb
import numpy as np

import genesis as gs
import genesis.utils.geom as gu

from uipc.core import Scene

if TYPE_CHECKING:
    from genesis.engine.entities import RigidEntity
    from genesis.engine.materials.rigid import CoupType


def has_articulation_dofs(entity: "RigidEntity") -> bool:
    """True if the entity has dynamic DOFs for IPC articulation coupling.

    Primitive morphs always keep a FIXED base joint (n_joints == 1, n_dofs == 0).
    Use n_dofs, not n_joints, so fixed Planes/Boxes are not treated as articulated.
    """
    return entity.n_dofs > 0


def default_coup_type(entity: "RigidEntity") -> "CoupType":
    """Auto-select coup_type when material.coup_type is None. See has_articulation_dofs."""
    if has_articulation_dofs(entity):
        if entity.base_link.is_fixed:
            return "external_articulation"
        return "two_way_soft_constraint"
    return "ipc_only"


def find_target_link_for_fixed_merge(link):
    """
    Find the target link for merging fixed joints.

    Walks up the kinematic tree, skipping links connected via FIXED joints, until finding a link with a non-FIXED joint
    or the root.

    This is similar to _merge_target_id in mjcf.py.

    Returns
    -------
    int
        The target link index to merge into
    """
    entity = link.entity

    while True:
        # If this is the root link (no parent), stop
        if link.parent_idx < 0:
            break

        # Check if there is any non-fixed joint
        if any(joint.type != gs.JOINT_TYPE.FIXED for joint in link.joints):
            # Found a link with non-FIXED joint, this is our target
            break

        # All joints are FIXED, move up to parent
        link = entity.links[link.parent_idx - entity.link_start]

    return link


def compute_link_to_link_transform(from_link, to_link):
    """
    Compute the relative transform from from_link to to_link.

    Returns
    -------
    tuple
        (pos, quat) transforming points from from_link frame to to_link frame
    """
    # Accumulate transforms going up from from_link to common ancestor (to_link)
    pos = np.array([0.0, 0.0, 0.0], dtype=gs.np_float)
    quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=gs.np_float)

    assert from_link.entity is to_link.entity
    entity = from_link.entity

    link = from_link
    while link is not to_link:
        if link.parent_idx < 0:
            gs.raise_exception(f"Cannot compute transform from link {from_link} to {to_link}")
        pos, quat = gu.transform_pos_quat_by_trans_quat(pos, quat, link.desc.pos, link.desc.quat)
        link = entity.links[link.parent_idx - entity.link_start]

    return pos, quat


def build_ipc_scene_config(options, sim_options, dt):
    """
    Build IPC Scene config dict from IPCCouplerOptions and SimOptions.

    Parameters
    ----------
    options : IPCCouplerOptions
        The coupler options
    sim_options : SimOptions
        The simulation options (provides gravity, requires_grad)
    dt : float
        The interval a single advance of the IPC world covers, which is the substep the coupler is called at.

    Returns
    -------
    dict
        Scene config dict ready to pass to Scene(config)
    """
    config = Scene.default_config()

    config["dt"] = dt
    gravity = sim_options.gravity
    config["gravity"] = [[float(e)] for e in gravity]

    # Newton solver options (only set if specified)
    _set_if_not_none(config, ["newton", "max_iter"], options.newton_max_iterations)
    _set_if_not_none(config, ["newton", "min_iter"], options.newton_min_iterations)
    _set_if_not_none(config, ["newton", "velocity_tol"], options.newton_tolerance)
    _set_if_not_none(config, ["newton", "ccd_tol"], options.newton_ccd_tolerance)
    _set_if_not_none(config, ["newton", "use_adaptive_tol"], options.newton_use_adaptive_tolerance)
    _set_if_not_none(config, ["newton", "transrate_tol"], options.newton_translation_tolerance)
    _set_if_not_none(config, ["newton", "semi_implicit", "enable"], options.newton_semi_implicit_enable)
    _set_if_not_none(config, ["newton", "semi_implicit", "beta_tol"], options.newton_semi_implicit_beta_tolerance)

    # Line search options
    _set_if_not_none(config, ["line_search", "max_iter"], options.n_linesearch_iterations)
    _set_if_not_none(config, ["line_search", "report_energy"], options.linesearch_report_energy)

    # Linear system options
    _set_if_not_none(config, ["linear_system", "solver"], options.linear_system_solver)
    _set_if_not_none(config, ["linear_system", "tol_rate"], options.linear_system_tolerance)

    # Contact options
    _set_if_not_none(config, ["contact", "enable"], options.contact_enable)
    _set_if_not_none(config, ["contact", "d_hat"], options.contact_d_hat)
    _set_if_not_none(config, ["contact", "friction", "enable"], options.contact_friction_enable)
    _set_if_not_none(config, ["contact", "eps_velocity"], options.contact_eps_velocity)
    _set_if_not_none(config, ["contact", "constitution"], options.contact_constitution)

    # Collision detection options
    _set_if_not_none(config, ["collision_detection", "method"], options.collision_detection_method)

    # CFL options
    _set_if_not_none(config, ["cfl", "enable"], options.cfl_enable)

    # Sanity check options
    _set_if_not_none(config, ["sanity_check", "enable"], options.sanity_check_enable)

    # Differential simulation options (derived from SimOptions)
    _set_if_not_none(config, ["diff_sim", "enable"], sim_options.requires_grad)

    return config


def _set_if_not_none(config, keys, value):
    """Set a nested config value only if it's not None."""
    if value is None:
        return
    # Cast to native Python types — UIPC pybind11 rejects numpy scalars and Python bool.
    # bool check must come before int since bool is a subclass of int.
    if isinstance(value, (bool, np.bool_)):
        value = int(value)
    elif isinstance(value, (int, np.integer)):
        value = int(value)
    elif isinstance(value, (float, np.floating)):
        value = float(value)
    d = config
    for key in keys[:-1]:
        d = d[key]
    d[keys[-1]] = value


def read_ipc_geometry_metadata(geo):
    """
    Read solver_type, env_idx, and entity/link index from IPC geometry metadata.

    Returns (solver_type, env_idx, idx) where idx is entity_idx for fem/cloth
    or link_idx for rigid. Returns None if the geometry has no solver_type
    metadata (i.e. not a Genesis-created geometry).
    """
    meta_attrs = geo.meta()
    solver_type_attr = meta_attrs.find("solver_type")
    if solver_type_attr is None:
        return None

    (solver_type,) = solver_type_attr.view()

    (env_idx,) = map(int, meta_attrs.find("env_idx").view())

    if solver_type == "rigid":
        (idx,) = map(int, meta_attrs.find("link_idx").view())
    elif solver_type in ("fem", "cloth"):
        (idx,) = map(int, meta_attrs.find("entity_idx").view())
    else:
        gs.raise_exception(f"Unknown IPC geometry solver_type: {solver_type!r}")

    return (solver_type, env_idx, idx)


@nb.jit(nopython=True, cache=True)
def update_coupling_forces(
    ipc_transforms,
    aim_transforms,
    links_mass,
    links_inertia_i,
    translation_strength,
    rotation_strength,
    out_forces,
    out_torques,
):
    """Compute coupling forces and torques for all links."""

    batch_shape = out_forces.shape[:-1]

    pos_current, R_current = ipc_transforms[..., :3, 3], ipc_transforms[..., :3, :3]
    pos_aim, R_aim = aim_transforms[..., :3, 3], aim_transforms[..., :3, :3]

    # Linear force
    out_forces[:] = (translation_strength * links_mass[..., None]) * (pos_current - pos_aim)

    # Relative rotation matrix
    R_rel = np.empty((*batch_shape, 3, 3), dtype=ipc_transforms.dtype)
    for idx in np.ndindex(batch_shape):
        R_rel[idx] = R_current[idx] @ R_aim[idx].T

    # Relative rotation in angle-axis representation
    rotvec = gu.R_to_rotvec(R_rel)

    # Transform inertia to world frame
    I_world = np.empty((*batch_shape, 3, 3), dtype=ipc_transforms.dtype)
    for idx in np.ndindex(batch_shape):
        I_world[idx] = R_current[idx] @ links_inertia_i[idx[-1:]] @ R_current[idx].T

    # Torque
    for idx in np.ndindex(batch_shape):
        out_torques[idx] = rotation_strength * (I_world[idx] @ rotvec[idx])
