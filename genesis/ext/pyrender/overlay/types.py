"""Static joint/entity metadata containers and helpers used by the overlay panel."""

from typing import TYPE_CHECKING, NamedTuple

import numpy as np

import genesis as gs

if TYPE_CHECKING:
    from genesis.engine.entities.rigid_entity import RigidEntity


QUATERNION_COMPONENT_LIMIT = 1.0


class EntityJointData(NamedTuple):
    """Joint metadata for an entity. ``q_limits`` is ``(lower, upper)``; ``quat_groups`` lists
    ``(start, end)`` index ranges of quaternion components for normalization."""

    q_names: list[str]
    q_limits: tuple[list[float], list[float]]
    q_is_quaternion: list[bool]
    quat_groups: list[tuple[int, int]]
    has_free_joint: bool
    free_joint_q_start: int


class EntityCacheEntry(NamedTuple):
    entity: "RigidEntity"
    name: str
    joint_data: EntityJointData
    n_qs: int
    n_dofs: int


def build_entity_joint_data(entity: "RigidEntity", free_joint_pos_limit: float) -> EntityJointData:
    """Build joint metadata for ``entity``. Handles free, spherical, and standard joints correctly."""
    q_names: list[str] = []
    q_limits_lower: list[float] = []
    q_limits_upper: list[float] = []
    q_is_quaternion: list[bool] = []
    quat_groups: list[tuple[int, int]] = []
    has_free_joint = False
    free_joint_q_start = -1

    for joint in entity.joints:
        if joint.n_qs == 0 or joint.type == gs.JOINT_TYPE.FIXED:
            continue

        if joint.type == gs.JOINT_TYPE.FREE:
            has_free_joint = True
            free_joint_q_start = len(q_names)
            q_names.extend(
                [
                    f"{joint.name}_x",
                    f"{joint.name}_y",
                    f"{joint.name}_z",
                    f"{joint.name}_qw",
                    f"{joint.name}_qx",
                    f"{joint.name}_qy",
                    f"{joint.name}_qz",
                ]
            )
            q_limits_lower.extend([-free_joint_pos_limit] * 3 + [-QUATERNION_COMPONENT_LIMIT] * 4)
            q_limits_upper.extend([free_joint_pos_limit] * 3 + [QUATERNION_COMPONENT_LIMIT] * 4)
            q_is_quaternion.extend([False, False, False, True, True, True, True])
            quat_groups.append((len(q_names) - 4, len(q_names)))
        elif joint.type == gs.JOINT_TYPE.SPHERICAL:
            quat_start = len(q_names)
            q_names.extend([f"{joint.name}_qw", f"{joint.name}_qx", f"{joint.name}_qy", f"{joint.name}_qz"])
            q_limits_lower.extend([-QUATERNION_COMPONENT_LIMIT] * 4)
            q_limits_upper.extend([QUATERNION_COMPONENT_LIMIT] * 4)
            q_is_quaternion.extend([True, True, True, True])
            quat_groups.append((quat_start, quat_start + 4))
        else:
            for i in range(joint.n_qs):
                name = joint.name if joint.n_qs == 1 else f"{joint.name}[{i}]"
                q_names.append(name)
                lo = float(joint.desc.dofs_limit[i, 0])
                hi = float(joint.desc.dofs_limit[i, 1])
                if not np.isfinite(lo):
                    lo = -1e6
                if not np.isfinite(hi):
                    hi = 1e6
                q_limits_lower.append(lo)
                q_limits_upper.append(hi)
                q_is_quaternion.append(False)

    return EntityJointData(
        q_names=q_names,
        q_limits=(q_limits_lower, q_limits_upper),
        q_is_quaternion=q_is_quaternion,
        quat_groups=quat_groups,
        has_free_joint=has_free_joint,
        free_joint_q_start=free_joint_q_start,
    )
