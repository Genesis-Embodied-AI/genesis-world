"""Pose helpers for controlling the center between the Franka fingertips."""

from __future__ import annotations

import numpy as np

import genesis.utils.geom as gu

# In the bundled Panda MJCF, each finger starts 0.0584 m above the hand
# origin and its contact pad is centered about 0.045 m farther along +Z.
# The standard Panda grasp-center convention rounds that transform to 0.1034 m.
PANDA_HAND_TO_GRIPPER_CENTER = np.array((0.0, 0.0, 0.1034), dtype=np.float64)


def gripper_center_from_hand_pose(
    hand_position: np.ndarray,
    hand_quaternion: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert the Panda hand-link pose to its fingertip-center pose."""
    quaternion = np.asarray(hand_quaternion, dtype=np.float64)
    position = np.asarray(hand_position, dtype=np.float64) + gu.transform_by_quat(
        PANDA_HAND_TO_GRIPPER_CENTER,
        quaternion,
    )
    return position, quaternion.copy()


def hand_pose_from_gripper_center(
    center_position: np.ndarray,
    center_quaternion: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a desired fingertip-center pose to the hand-link IK target."""
    quaternion = np.asarray(center_quaternion, dtype=np.float64)
    position = np.asarray(center_position, dtype=np.float64) - gu.transform_by_quat(
        PANDA_HAND_TO_GRIPPER_CENTER,
        quaternion,
    )
    return position, quaternion.copy()
