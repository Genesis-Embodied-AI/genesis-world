from typing import Dict, List, NamedTuple, Tuple

import numpy as np
from pxr import Usd, UsdPhysics

import genesis as gs
from genesis.utils import geom as gu
from genesis.utils import urdf as urdf_utils

from .usd_context import (
    UsdContext,
    extract_links_referenced_by_joints,
    find_joints_in_range,
    find_rigid_bodies_in_range,
    get_current_usd_context,
    resolve_rigid_body_link_path,
)
from .usd_collision import apply_collision_filtering
from .usd_geometry import parse_prim_geoms
from .usd_utils import (
    AXES_VECTOR,
    get_attr_value_by_candidates,
    usd_center_of_mass_to_numpy,
    usd_density_to_float,
    usd_inertia_to_numpy,
    usd_mass_to_float,
    usd_pos_to_numpy,
    usd_principal_axes_to_numpy,
    usd_quat_to_numpy,
)


DRIVE_NAMES = {
    gs.JOINT_TYPE.REVOLUTE: (("angular",),),
    gs.JOINT_TYPE.PRISMATIC: (("linear",),),
    gs.JOINT_TYPE.SPHERICAL: (
        ("rotX",),
        ("rotY",),
        ("rotZ",),
    ),
    gs.JOINT_TYPE.FIXED: (),
    gs.JOINT_TYPE.FREE: (
        ("transX",),
        ("transY",),
        ("transZ",),
        ("rotX",),
        ("rotY",),
        ("rotZ",),
    ),
}


_DOF_NAMES = ("transX", "transY", "transZ", "rotX", "rotY", "rotZ")
_ROT_DOFS = {"rotX", "rotY", "rotZ"}
_DOF_AXIS = {"transX": "X", "transY": "Y", "transZ": "Z", "rotX": "X", "rotY": "Y", "rotZ": "Z"}


class _LinkJoint(NamedTuple):
    """A single joint incident on a link, as consumed by ``_parse_link``.

    Attributes
    ----------
    joint_prim : Usd.Prim | None
        The joint prim connecting this link to its parent, or ``None`` for a link with no joint
        (a floating/fixed root that gets an implicit free/fixed joint).
    parent_idx : int
        Index of the parent link in the entity's link list, or ``-1`` for a root link.
    is_body1 : bool
        Whether this link is the joint's ``body1`` (child) rather than ``body0`` (parent). Selects
        which ``localPos``/``localRot`` of the joint describes this link's anchor.
    frame_path : str | None
        Path of the prim that the joint's ``body0``/``body1`` relationship for this link literally
        targets. USD authors ``localPos``/``localRot`` in this prim's frame, which may be a
        collision child or wrapper xform rather than the canonical RigidBodyAPI link. ``None`` when
        the relationship has no target (e.g. jointed to the world).
    """

    joint_prim: Usd.Prim | None
    parent_idx: int
    is_body1: bool
    frame_path: str | None


def _detect_generic_joint_type(joint_prim: Usd.Prim) -> Tuple[int, int, int, str | None]:
    """Detect Genesis joint type for a generic PhysicsJoint by inspecting DOF limits and drives.

    If no DriveAPI or LimitAPI is applied to any DOF, the joint is treated as FREE (6 DOF).
    Otherwise, a DOF is enabled if DriveAPI is applied, or LimitAPI is applied with low < high.
    A DOF with LimitAPI where low >= high (e.g., [1.0, -1.0]) is locked.

    Returns (joint_type, n_dofs, n_qs, detected_axis_str).
    """
    has_any_config = False
    enabled_dofs = set()
    for dof_name in _DOF_NAMES:
        if joint_prim.HasAPI(UsdPhysics.DriveAPI, dof_name):
            has_any_config = True
            enabled_dofs.add(dof_name)
        elif joint_prim.HasAPI(UsdPhysics.LimitAPI, dof_name):
            has_any_config = True
            limit = UsdPhysics.LimitAPI.Get(joint_prim, dof_name)
            low, high = limit.GetLowAttr().Get(), limit.GetHighAttr().Get()
            if low is None or high is None or low < high:
                enabled_dofs.add(dof_name)

    # No DOF configuration at all → treat as FREE (all DOFs enabled)
    if not has_any_config:
        return gs.JOINT_TYPE.FREE, 6, 7, None

    rot_dofs = enabled_dofs & _ROT_DOFS
    trans_dofs = enabled_dofs - _ROT_DOFS

    if not enabled_dofs:
        return gs.JOINT_TYPE.FIXED, 0, 0, None
    if not trans_dofs:
        if len(rot_dofs) == 1:
            return gs.JOINT_TYPE.REVOLUTE, 1, 1, _DOF_AXIS[next(iter(rot_dofs))]
        return gs.JOINT_TYPE.SPHERICAL, 3, 4, None
    if not rot_dofs and len(trans_dofs) == 1:
        return gs.JOINT_TYPE.PRISMATIC, 1, 1, _DOF_AXIS[next(iter(trans_dofs))]
    return gs.JOINT_TYPE.FREE, 6, 7, None


def _parse_joint_axis_pos(
    context: UsdContext,
    joint: UsdPhysics.Joint,
    child_link: Usd.Prim,
    joint_frame: Usd.Prim,
    is_body1: bool,
    axis_override: str | None = None,
) -> Tuple[str, np.ndarray, np.ndarray]:
    # localPos/localRot are expressed in the frame of 'joint_frame', the joint body-relationship target,
    # which may be a collision child prim rather than the canonical RigidBodyAPI link prim.
    joint_pos_attr = joint.GetLocalPos1Attr() if is_body1 else joint.GetLocalPos0Attr()
    joint_pos = usd_pos_to_numpy(joint_pos_attr.Get()) if joint_pos_attr.HasValue() else gu.zero_pos()
    T = context.compute_transform(joint_frame)
    joint_pos = gu.transform_by_T(joint_pos, T)
    Q, S = context.compute_gs_transform(child_link)
    Q_inv = np.linalg.inv(Q)
    joint_pos = Q_inv[:3, :3] @ (joint_pos - Q[:3, 3])

    if isinstance(joint, (UsdPhysics.PrismaticJoint, UsdPhysics.RevoluteJoint)):
        joint_quat = usd_quat_to_numpy((joint.GetLocalRot1Attr() if is_body1 else joint.GetLocalRot0Attr()).Get())
        joint_axis_str = joint.GetAxisAttr().Get() or "X"
        joint_axis = gu.transform_by_quat(AXES_VECTOR[joint_axis_str], joint_quat)
        joint_axis = gu.transform_by_R(joint_axis, T[:3, :3])
        joint_axis = Q_inv[:3, :3] @ joint_axis
        if np.linalg.norm(joint_axis) < gs.EPS:
            gs.raise_exception(f"Joint axis is zero for joint {joint.GetPath()}.")
        joint_axis /= np.linalg.norm(joint_axis)
    elif axis_override is not None:
        joint_quat = usd_quat_to_numpy((joint.GetLocalRot1Attr() if is_body1 else joint.GetLocalRot0Attr()).Get())
        joint_axis = gu.transform_by_quat(AXES_VECTOR[axis_override], joint_quat)
        joint_axis = gu.transform_by_R(joint_axis, T[:3, :3])
        joint_axis = Q_inv[:3, :3] @ joint_axis
        if np.linalg.norm(joint_axis) < gs.EPS:
            gs.raise_exception(f"Joint axis is zero for joint {joint.GetPath()}.")
        joint_axis /= np.linalg.norm(joint_axis)
        joint_axis_str = axis_override
    else:
        joint_axis_str, joint_axis = None, None

    return joint_axis_str, joint_axis, joint_pos


def _parse_link(
    context: UsdContext,
    link: Usd.Prim,
    joints: List[_LinkJoint],
    links: List[Usd.Prim],
    morph: gs.morphs.USD,
):
    l_info = {}
    l_info["name"] = str(link.GetPath())
    l_info["invweight"] = np.full((2,), fill_value=-1.0)

    # parse link fixed state
    link_fixed = False
    if link.HasAPI(UsdPhysics.RigidBodyAPI):
        rigid_body_api = UsdPhysics.RigidBodyAPI(link)
        if rigid_body_api.GetKinematicEnabledAttr().Get():
            link_fixed = True
        elif rigid_body_api.GetRigidBodyEnabledAttr().Get() is False:
            link_fixed = True
    elif link.HasAPI(UsdPhysics.CollisionAPI):
        link_fixed = True

    # Apply morph.fixed override for root links
    is_root = any(joint.parent_idx == -1 for joint in joints)
    if morph.fixed is not None and is_root:
        link_fixed = morph.fixed

    # parse link mass properties
    if link.HasAPI(UsdPhysics.MassAPI):
        mass_api = UsdPhysics.MassAPI(link)

        com_attr = mass_api.GetCenterOfMassAttr()
        l_info["inertial_pos"] = usd_center_of_mass_to_numpy(com_attr.Get())
        principal_axes_attr = mass_api.GetPrincipalAxesAttr()
        l_info["inertial_quat"] = usd_principal_axes_to_numpy(principal_axes_attr.Get())
        inertia_attr = mass_api.GetDiagonalInertiaAttr()
        l_info["inertial_i"] = usd_inertia_to_numpy(inertia_attr.Get())
        mass_attr = mass_api.GetMassAttr()
        l_info["inertial_mass"] = usd_mass_to_float(mass_attr.Get())

    # set link transform for pure rigid bodies (no joints)
    if not joints:
        Q, S = context.compute_gs_transform(link, None)
        l_info["parent_idx"] = -1
        l_info["pos"] = Q[:3, 3]
        l_info["quat"] = gu.R_to_quat(Q[:3, :3])

    j_infos = []
    for joint_prim, parent_idx, is_body1, frame_path in joints:
        if "parent_idx" not in l_info:
            parent_link = None if parent_idx == -1 else links[parent_idx]
            Q, S = context.compute_gs_transform(link, parent_link)
            l_info["parent_idx"] = parent_idx
            l_info["pos"] = Q[:3, 3]
            l_info["quat"] = gu.R_to_quat(Q[:3, :3])

        elif l_info["parent_idx"] != parent_idx:
            gs.raise_exception(f"Link {link.GetPath()} has multiple parents: {l_info['parent_idx']} and {parent_idx}.")

        if joint_prim is not None:
            joint_type = gs.JOINT_TYPE.FIXED
            n_dofs, n_qs = 0, 0
            detected_axis_str = None
            if not link_fixed:
                if joint_prim.IsA(UsdPhysics.RevoluteJoint):
                    joint_type = gs.JOINT_TYPE.REVOLUTE
                    joint = UsdPhysics.RevoluteJoint(joint_prim)
                    n_dofs, n_qs = 1, 1
                elif joint_prim.IsA(UsdPhysics.PrismaticJoint):
                    joint_type = gs.JOINT_TYPE.PRISMATIC
                    joint = UsdPhysics.PrismaticJoint(joint_prim)
                    n_dofs, n_qs = 1, 1
                elif joint_prim.IsA(UsdPhysics.SphericalJoint):
                    joint_type = gs.JOINT_TYPE.SPHERICAL
                    joint = UsdPhysics.SphericalJoint(joint_prim)
                    n_dofs, n_qs = 3, 4
                elif joint_prim.GetTypeName() == "PhysicsJoint":
                    joint_type, n_dofs, n_qs, detected_axis_str = _detect_generic_joint_type(joint_prim)
                    joint = UsdPhysics.Joint(joint_prim)
                elif not joint_prim.IsA(UsdPhysics.FixedJoint):
                    gs.logger.warning(
                        f"Unsupported USD joint type: {joint_prim.GetTypeName()} for {joint_prim.GetPath()}. "
                        "Parsed as fixed joint."
                    )
            if joint_type == gs.JOINT_TYPE.FIXED:
                joint = UsdPhysics.Joint(joint_prim)
            # localPos/localRot are authored in the body-relationship target's frame (frame_path),
            # which may differ from the canonical link prim; fall back to the link if unresolved.
            joint_frame = context.stage.GetPrimAtPath(frame_path) if frame_path else link
            if not joint_frame.IsValid():
                joint_frame = link
            joint_axis_str, joint_axis, joint_pos = _parse_joint_axis_pos(
                context, joint, link, joint_frame, is_body1, detected_axis_str
            )
            joint_name = str(joint_prim.GetPath())
        else:
            if link_fixed:
                joint_type = gs.JOINT_TYPE.FIXED
                n_dofs, n_qs = 0, 0
            else:
                joint_type = gs.JOINT_TYPE.FREE
                n_dofs, n_qs = 6, 7
            joint = None
            joint_axis_str, joint_axis, joint_pos = None, None, gu.zero_pos()
            joint_name = f"{l_info['name']}_joint"

        j_info = {
            "name": joint_name,
            "sol_params": gu.default_solver_params(),
            "n_qs": n_qs,
            "n_dofs": n_dofs,
            "type": joint_type,
            "pos": joint_pos,
            "dofs_invweight": np.full(n_dofs, -1.0, dtype=gs.np_float),
        }

        if joint_type in (gs.JOINT_TYPE.REVOLUTE, gs.JOINT_TYPE.PRISMATIC):
            # TODO: use attribute "state:<INSTANCE_NAME>:physics:positiion" to parse init_qpos (but it is IsaacSim specific)
            j_info["init_qpos"] = np.zeros(n_qs, dtype=gs.np_float)

            if joint_type == gs.JOINT_TYPE.REVOLUTE:
                j_info["dofs_motion_ang"] = joint_axis[None]
                j_info["dofs_motion_vel"] = np.zeros((1, 3), dtype=gs.np_float)
                j_info["dofs_stiffness"] = np.array(
                    [
                        get_attr_value_by_candidates(
                            joint_prim,
                            candidates=morph.revolute_joint_stiffness_attr_candidates,
                            attr_name="dofs_stiffness",
                            default_value=0.0,
                        )
                    ],
                    dtype=gs.np_float,
                )
                j_info["dofs_damping"] = np.array(
                    [
                        get_attr_value_by_candidates(
                            joint_prim,
                            candidates=morph.revolute_joint_damping_attr_candidates,
                            attr_name="dofs_damping",
                            default_value=0.0,
                        )
                    ],
                    dtype=gs.np_float,
                )
                # NOTE: No idea how to scale the angle limits under non-uniform scaling now.
                # For generic PhysicsJoint detected as REVOLUTE, use LimitAPI instead of joint attrs
                if detected_axis_str is not None:
                    dof_name = f"rot{detected_axis_str}"
                    lower_limit, upper_limit = -np.inf, np.inf
                    if joint_prim.HasAPI(UsdPhysics.LimitAPI, dof_name):
                        limit = UsdPhysics.LimitAPI.Get(joint_prim, dof_name)
                        low, high = limit.GetLowAttr().Get(), limit.GetHighAttr().Get()
                        if low is not None:
                            lower_limit = np.deg2rad(low)
                        if high is not None:
                            upper_limit = np.deg2rad(high)
                else:
                    lower_limit_attr = joint.GetLowerLimitAttr()
                    upper_limit_attr = joint.GetUpperLimitAttr()
                    lower_limit = np.deg2rad(lower_limit_attr.Get()) if lower_limit_attr.HasValue() else -np.inf
                    upper_limit = np.deg2rad(upper_limit_attr.Get()) if upper_limit_attr.HasValue() else np.inf
                j_info["dofs_limit"] = np.asarray([[lower_limit, upper_limit]], dtype=gs.np_float)
            else:  # joint_type == gs.JOINT_TYPE.PRISMATIC
                j_info["dofs_motion_ang"] = np.zeros((1, 3), dtype=gs.np_float)
                j_info["dofs_motion_vel"] = joint_axis[None]
                j_info["dofs_stiffness"] = np.array(
                    [
                        get_attr_value_by_candidates(
                            joint_prim,
                            candidates=morph.prismatic_joint_stiffness_attr_candidates,
                            attr_name="dofs_stiffness",
                            default_value=0.0,
                        )
                    ],
                    dtype=gs.np_float,
                )
                j_info["dofs_damping"] = np.array(
                    [
                        get_attr_value_by_candidates(
                            joint_prim,
                            candidates=morph.prismatic_joint_damping_attr_candidates,
                            attr_name="dofs_damping",
                            default_value=0.0,
                        )
                    ],
                    dtype=gs.np_float,
                )
                # For generic PhysicsJoint detected as PRISMATIC, use LimitAPI
                if detected_axis_str is not None:
                    dof_name = f"trans{detected_axis_str}"
                    lower_limit, upper_limit = -np.inf, np.inf
                    if joint_prim.HasAPI(UsdPhysics.LimitAPI, dof_name):
                        limit = UsdPhysics.LimitAPI.Get(joint_prim, dof_name)
                        low, high = limit.GetLowAttr().Get(), limit.GetHighAttr().Get()
                        if low is not None:
                            lower_limit = float(low)
                        if high is not None:
                            upper_limit = float(high)
                else:
                    lower_limit_attr = joint.GetLowerLimitAttr()
                    upper_limit_attr = joint.GetUpperLimitAttr()
                    lower_limit = lower_limit_attr.Get() if lower_limit_attr.HasValue() else -np.inf
                    upper_limit = upper_limit_attr.Get() if upper_limit_attr.HasValue() else np.inf
                j_info["dofs_limit"] = np.asarray([[lower_limit, upper_limit]], dtype=gs.np_float) * morph.scale
                j_info["init_qpos"] *= morph.scale

        else:
            j_info["dofs_stiffness"] = np.zeros(n_dofs, dtype=gs.np_float)
            j_info["dofs_damping"] = np.zeros(n_dofs, dtype=gs.np_float)

            if joint_type == gs.JOINT_TYPE.SPHERICAL:
                j_info["dofs_motion_ang"] = np.eye(3)
                j_info["dofs_motion_vel"] = np.zeros((3, 3))
                j_info["init_qpos"] = gu.identity_quat()
                # Native SphericalJoint uses cone limits (ConeAngle0/1), while generic PhysicsJoint
                # (D6) uses per-axis pyramid limits via LimitAPI:rotX/rotY/rotZ. Neither is currently
                # enforced by the Genesis solver for spherical joints.
                # Ref: https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/rigid_bodies_articulations/joints.html#spherical-joint
                dofs_limit = []
                for dof_name in ("rotX", "rotY", "rotZ"):
                    if joint_prim is not None and joint_prim.HasAPI(UsdPhysics.LimitAPI, dof_name):
                        limit = UsdPhysics.LimitAPI.Get(joint_prim, dof_name)
                        low, high = limit.GetLowAttr().Get(), limit.GetHighAttr().Get()
                        lo = np.deg2rad(low) if low is not None and low < high else -np.inf
                        hi = np.deg2rad(high) if high is not None and low < high else np.inf
                        dofs_limit.append([lo, hi])
                    else:
                        dofs_limit.append([-np.inf, np.inf])
                j_info["dofs_limit"] = np.asarray(dofs_limit, dtype=gs.np_float)
            elif joint_type == gs.JOINT_TYPE.FIXED:
                j_info["dofs_motion_ang"] = np.zeros((0, 3))
                j_info["dofs_motion_vel"] = np.zeros((0, 3))
                j_info["dofs_limit"] = np.zeros((0, 2))
                j_info["init_qpos"] = np.zeros(0)
            else:  # joint_type == gs.JOINT_TYPE.FREE
                j_info["dofs_motion_ang"] = np.eye(6, 3, -3)
                j_info["dofs_motion_vel"] = np.eye(6, 3)
                j_info["dofs_limit"] = np.tile([-np.inf, np.inf], (6, 1))
                j_info["init_qpos"] = np.concatenate([l_info["pos"] * morph.scale, l_info["quat"]])

        # Only parse joint dynamics and drive API for non-fixed and non-free joints
        if joint_type not in (gs.JOINT_TYPE.FIXED, gs.JOINT_TYPE.FREE):
            j_info["dofs_frictionloss"] = np.full(
                (n_dofs,),
                get_attr_value_by_candidates(
                    joint_prim,
                    candidates=morph.joint_friction_attr_candidates,
                    attr_name="dofs_frictionloss",
                    default_value=0.0,
                ),
                dtype=gs.np_float,
            )
            j_info["dofs_armature"] = np.full(
                (n_dofs,),
                get_attr_value_by_candidates(
                    joint_prim,
                    candidates=morph.joint_armature_attr_candidates,
                    attr_name="dofs_armature",
                    default_value=0.0,
                ),
                dtype=gs.np_float,
            )

            # parse drive API
            # References: https://openusd.org/release/api/class_usd_physics_drive_a_p_i.html
            # Note: dofs_stiffness and dofs_damping are NOT set here - they are passive joint properties
            # that come from joint dynamics, not from DriveAPI (which is an active control system)
            dofs_kp, dofs_kv, dofs_force_range = [], [], []

            for drive_components in DRIVE_NAMES[joint_type]:
                dof_kp, dof_kv, max_force = 0.0, 0.0, np.inf
                if joint_axis_str:
                    drive_components = drive_components + (joint_axis_str,)
                for drive_component in drive_components:
                    if joint_prim.HasAPI(UsdPhysics.DriveAPI, drive_component):
                        drive = UsdPhysics.DriveAPI.Get(joint_prim, drive_component)
                        # TODO: use drive.GetTypeAttr().Get() to parse force or velocity.
                        # Note: Defaults are 0 (stiffness/damping) and inf (maxForce), which are valid.
                        # Using 'or' is safe here since fallback values match the defaults.
                        dof_kp = drive.GetStiffnessAttr().Get() or dof_kp
                        dof_kv = drive.GetDampingAttr().Get() or dof_kv
                        max_force = drive.GetMaxForceAttr().Get() or max_force
                        break

                dofs_kp.append(dof_kp)
                dofs_kv.append(dof_kv)
                dofs_force_range.append([-max_force, max_force])

            # TODO: Implement target solving in rigid solver. (GetTargetPositionAttr())
            kp = np.asarray(dofs_kp, dtype=gs.np_float)
            kv = np.asarray(dofs_kv, dtype=gs.np_float)
            j_info["dofs_act_gain"] = kp
            j_info["dofs_act_bias"] = np.column_stack([np.zeros_like(kp), -kp, -kv])
            j_info["dofs_force_range"] = np.asarray(dofs_force_range, dtype=gs.np_float)
        else:
            j_info["dofs_frictionloss"] = np.zeros(n_dofs, dtype=gs.np_float)
            j_info["dofs_armature"] = np.zeros(n_dofs, dtype=gs.np_float)
            j_info["dofs_act_gain"] = np.zeros(n_dofs, dtype=gs.np_float)
            j_info["dofs_act_bias"] = np.zeros((n_dofs, 3), dtype=gs.np_float)
            j_info["dofs_force_range"] = np.tile([-np.inf, np.inf], (n_dofs, 1))

        j_infos.append(j_info)

    if abs(1.0 - morph.scale) > gs.EPS:
        l_info["pos"] *= morph.scale
        if l_info.get("inertial_pos") is not None:
            l_info["inertial_pos"] *= morph.scale
        if l_info.get("inertial_mass") is not None:
            l_info["inertial_mass"] *= morph.scale**3
        if l_info.get("inertial_i") is not None:
            l_info["inertial_i"] *= morph.scale**5
        l_info["invweight"][:] = -1.0
        for j_info in j_infos:
            j_info["pos"] *= morph.scale
            # TODO: parse actuator in USD articulation, now all joints are considered to have actuators
            j_info["dofs_act_gain"] *= morph.scale**3
            j_info["dofs_act_bias"] *= morph.scale**3
            j_info["dofs_invweight"][:] = -1.0

    return l_info, j_infos


# Rigidbody requirements: https://docs.omniverse.nvidia.com/kit/docs/asset-requirements/latest/capabilities/physics_bodies/physics_rigid_bodies/capability-physics_rigid_bodies.html
# Joint requirements: https://docs.omniverse.nvidia.com/kit/docs/asset-requirements/latest/capabilities/physics_bodies/physics_joints/capability-physics_joints.html
def _parse_articulation_structure(stage: Usd.Stage, entity_prim: Usd.Prim, joint_prims: List[str] | None = None):
    link_path_joints = {}

    # Get joints to process - joint_prims is computed by parse_usd_rigid_entity (from the stage,
    # as joints that reference links in the entity subtree) and passed here.
    # If None, it means no joints (pure rigid body case).
    if joint_prims is not None:
        # Use provided joint prims
        joint_prim_objs = [stage.GetPrimAtPath(joint_path) for joint_path in joint_prims]
        for prim in joint_prim_objs:
            if not prim.IsValid():
                gs.raise_exception(f"Invalid joint prim path: {prim}")
            if not prim.IsA(UsdPhysics.Joint):
                gs.raise_exception(f"Prim {prim.GetPath()} is not a joint.")
    else:
        # No joints provided - this is a pure rigid body case
        joint_prim_objs = []

    for prim in joint_prim_objs:
        joint = UsdPhysics.Joint(prim)
        body0_targets = joint.GetBody0Rel().GetTargets()  # parent
        body1_targets = joint.GetBody1Rel().GetTargets()  # child
        body0_original_path = str(body0_targets[0]) if body0_targets else None
        body1_original_path = str(body1_targets[0]) if body1_targets else None
        body0_target_path = resolve_rigid_body_link_path(stage, body0_original_path) if body0_original_path else None
        body1_target_path = resolve_rigid_body_link_path(stage, body1_original_path) if body1_original_path else None

        # Only process if at least one body is a rigid body
        body0_is_rigid = body0_target_path is not None
        body1_is_rigid = body1_target_path is not None

        # Intermediate tuples carry the parent *path* (resolved to a link index in the pass below);
        # frame_path is this link's own original relationship target (body1 for a child, body0 else).
        if body1_is_rigid:
            parent_path = body0_target_path if body0_is_rigid else None
            link_path_joints.setdefault(body1_target_path, []).append((prim, parent_path, True, body1_original_path))
            # If body0 is also a rigid body, add it as a link (may be parent)
            if body0_is_rigid:
                link_path_joints.setdefault(body0_target_path, [])
        elif body0_is_rigid:
            link_path_joints.setdefault(body0_target_path, []).append((prim, None, False, body0_original_path))
        # If neither is a rigid body, skip this joint (it doesn't connect rigid bodies)

    links, link_joints = [], []
    if link_path_joints:
        link_path_to_idx = {link_path: idx for idx, link_path in enumerate(link_path_joints.keys())}
        link_path_to_idx[None] = -1
        for link_path, joints in link_path_joints.items():
            if not stage.GetPrimAtPath(link_path):
                gs.raise_exception(f"Link {link_path} not found in stage.")
            links.append(stage.GetPrimAtPath(link_path))
            link_joints.append(
                [
                    _LinkJoint(joint, link_path_to_idx[parent_path], is_body1, frame_path)
                    for joint, parent_path, is_body1, frame_path in joints
                ]
            )
    else:
        # Pure rigid body case - no joints, no placeholder needed
        links = [entity_prim]
        link_joints = [[]]
        link_path_to_idx = {None: -1, str(entity_prim.GetPath()): 0}
    for joints in link_joints:
        if not joints:
            joints.append(_LinkJoint(None, -1, False, None))

    return links, link_joints, link_path_to_idx


def _parse_geoms(
    context: UsdContext,
    entity_prim: Usd.Prim,
    link_path_to_idx: Dict[str, int],
    morph: gs.morphs.USD,
    surface: gs.surfaces.Surface,
) -> List[List[Dict]]:
    links_g_infos = [[] for _ in range(len(link_path_to_idx))]
    parse_prim_geoms(context, entity_prim, None, links_g_infos, link_path_to_idx, morph, surface)
    return links_g_infos


def _parse_links(
    context: UsdContext,
    links: List[Usd.Prim],
    link_joints: List[List[_LinkJoint]],
    morph: gs.morphs.USD,
) -> Tuple[List[Dict], List[List[Dict]]]:
    l_infos = []
    links_j_infos = []
    for link, joints in zip(links, link_joints):
        l_info, link_j_info = _parse_link(context, link, joints, links, morph)
        l_infos.append(l_info)
        links_j_infos.append(link_j_info)
    return l_infos, links_j_infos


def _compute_joint_prim_paths(stage: Usd.Stage, entity_prim: Usd.Prim) -> List[str] | None:
    """
    Compute joint prim paths for an entity. Joints are those that reference (body0 or body1)
    any rigid body in the entity subtree, so that entities whose prim_path is a link (e.g.
    common ancestor of a single link) still get joints that are siblings of that link.

    This determines whether the entity is:
    - A pure rigid body (no joints referencing links in subtree)
    - A pure articulation (has joints, all rigid bodies in subtree are referenced by them)
    - Mixed case (error: has both joints and unreferenced rigid bodies in subtree)

    Raises
    ------
    gs.GenesisException
        If mixed entity is detected (both joints and unreferenced rigid bodies in subtree).
    """
    # Find all rigid bodies in the entity_prim subtree
    rigid_bodies_in_subtree = find_rigid_bodies_in_range(Usd.PrimRange(entity_prim))

    # Find all joints in the stage (filter to those referencing links in subtree below)
    all_joints = find_joints_in_range(Usd.PrimRange(stage.GetPseudoRoot()))

    # Joints belonging to this entity: those that reference any link in the entity subtree
    joints_for_entity = []
    for joint_prim in all_joints:
        joint = UsdPhysics.Joint(joint_prim)
        body0_targets = joint.GetBody0Rel().GetTargets()
        body1_targets = joint.GetBody1Rel().GetTargets()
        body0_path = str(body0_targets[0]) if body0_targets else None
        body1_path = str(body1_targets[0]) if body1_targets else None
        body0_resolved = resolve_rigid_body_link_path(stage, body0_path) if body0_path else None
        body1_resolved = resolve_rigid_body_link_path(stage, body1_path) if body1_path else None
        if (body0_resolved and body0_resolved in rigid_bodies_in_subtree) or (
            body1_resolved and body1_resolved in rigid_bodies_in_subtree
        ):
            joints_for_entity.append(joint_prim)

    links_referenced_by_joints = extract_links_referenced_by_joints(stage, joints_for_entity, check_rigid_body=False)

    # Determine entity type (has joints vs pure rigid body, and mixed-case check)
    has_joints = len(joints_for_entity) > 0
    has_unreferenced_rigid_bodies = len(rigid_bodies_in_subtree - links_referenced_by_joints) > 0

    # Check for mixed case error, because scene.add_entity(...) only return 1 entity.
    if has_joints and has_unreferenced_rigid_bodies:
        unreferenced = rigid_bodies_in_subtree - links_referenced_by_joints
        gs.raise_exception(
            f"Mixed entity detected at {entity_prim.GetPath()}: "
            f"has {len(joints_for_entity)} joints but also has {len(unreferenced)} rigid bodies "
            f"not referenced by joints: {list(unreferenced)[:5]}. "
            "Use scene.add_stage() to handle mixed entities, or ensure all rigid bodies are connected by joints."
        )

    # Pure articulation case (has joints, all rigid bodies are referenced)
    if has_joints:
        return [str(joint.GetPath()) for joint in joints_for_entity]

    # Pure rigid body case (no joints)
    return None


def parse_usd_rigid_entity(morph: gs.morphs.USD, surface: gs.surfaces.Surface):
    """
    Unified parser for USD rigid entities (both articulations and rigid bodies).

    Treats rigid bodies as articulation roots with no child links.
    Automatically detects whether the prim is an articulation (has joints) or
    a rigid body (no joints) and processes accordingly.

    Parameters
    ----------
    morph : gs.morphs.USD
        USD morph configuration.
    surface : gs.surfaces.Surface
        Surface configuration.

    Returns
    -------
    l_infos : list
        List of link info dictionaries.
    links_j_infos : list
        List of lists of joint info dictionaries.
    links_g_infos : list
        List of lists of geometry info dictionaries.
    eqs_info : list
        List of equality constraint info dictionaries.
    """
    context = get_current_usd_context() or UsdContext(morph.file)
    context.find_all_materials()
    stage: Usd.Stage = context.stage

    if morph.prim_path is None:
        gs.logger.debug("USD morph has no prim path. Fallback to its default prim path.")
        entity_prim = stage.GetDefaultPrim()
    else:
        entity_prim = stage.GetPrimAtPath(morph.prim_path)
    if not entity_prim.IsValid():
        if morph.prim_path is None:
            err_msg = (
                f"Invalid default prim path {entity_prim} in USD file {morph.file}. Please specify 'morph.prim_path'."
            )
        else:
            err_msg = f"Invalid user-specified prim path {entity_prim} in USD file {morph.file}."
        gs.raise_exception(err_msg)

    # Deduce joint prim paths for this entity and parse articulation structure (links + joints)
    joint_prims = _compute_joint_prim_paths(stage, entity_prim)
    links, link_joints, link_path_to_idx = _parse_articulation_structure(stage, entity_prim, joint_prims)
    links_g_infos = _parse_geoms(context, entity_prim, link_path_to_idx, morph, surface)

    # Visual fallback: duplicate collision geometry as visual when no visual geometry exists,
    # following the same pattern as MJCF (genesis/utils/mjcf.py).
    if surface.texture is None:
        surface = surface.model_copy()
        surface.update_texture()
    for link_g_infos in links_g_infos:
        has_visual = any("vmesh" in g for g in link_g_infos)
        if has_visual:
            continue
        collision_entries = [g for g in link_g_infos if g.get("contype") or g.get("conaffinity")]
        if collision_entries:
            gs.logger.warning(
                "No visual geometry found for link. Falling back to using collision geometries as visual."
            )
        for g_info in collision_entries:
            g_info = g_info.copy()
            mesh = g_info.pop("mesh")
            mesh = gs.Mesh.from_trimesh(mesh=mesh.trimesh, surface=surface, metadata=mesh.metadata)
            g_info = {**g_info, "vmesh": mesh, "contype": 0, "conaffinity": 0}
            link_g_infos.append(g_info)

    # Apply USD collision filtering (CollisionGroup / FilteredPairsAPI) to this entity's collision geoms.
    cg_infos = [g for link_g_infos in links_g_infos for g in link_g_infos if g.get("contype") or g.get("conaffinity")]
    apply_collision_filtering(context, cg_infos)

    l_infos, links_j_infos = _parse_links(context, links, link_joints, morph)

    # A density authored on the link's MassAPI overrides any per-geom physics-material density (USD
    # precedence); the per-geom densities then drive the rigid build's link inertial estimate wherever
    # no explicit mass is authored.
    for link, link_g_infos in zip(links, links_g_infos):
        if not link.HasAPI(UsdPhysics.MassAPI):
            continue
        link_density = usd_density_to_float(UsdPhysics.MassAPI(link).GetDensityAttr().Get())
        if link_density is None:
            continue
        for g_info in link_g_infos:
            if g_info.get("contype") or g_info.get("conaffinity"):
                g_info["density"] = link_density

    # The prim paths are fully consumed at this point; strip them so they do not leak into the generic
    # entity-building pipeline (including the visual copies of collision geoms).
    for link_g_infos in links_g_infos:
        for g_info in link_g_infos:
            g_info.pop("prim_path", None)

    l_infos, links_j_infos, links_g_infos, _ = urdf_utils.order_links_depth_first(l_infos, links_j_infos, links_g_infos)
    eqs_info = []  # USD doesn't support equality constraints

    return l_infos, links_j_infos, links_g_infos, eqs_info
