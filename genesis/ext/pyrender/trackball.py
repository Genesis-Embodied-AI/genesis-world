"""Trackball moving the camera of the viewer with the mouse."""

from enum import IntEnum

import numpy as np

import trimesh.transformations as transformations


EPSILON = np.finfo(np.float32).eps


class TrackballMode(IntEnum):
    """The motion that dragging the mouse gives the camera."""

    ROTATE = 0
    """Orbit about the target."""
    PAN = 1
    """Move along the image plane, with the target."""
    ZOOM = 2
    """Move towards the target or away from it."""


class Trackball:
    """The camera of the viewer, orbiting about a target point as the mouse drags it.

    A drag moves the camera from its pose at the start of the drag.

    Parameters
    ----------
    pose : (4, 4) float
        The initial camera-to-world pose.
    size : (2,) float
        The width and height of the image of the camera, in pixels.
    scale : float
        The length that a pan across the image moves the camera by, in meters.
    target : (3,) float
        The point that the camera orbits about.
    """

    def __init__(self, pose, size, scale, target):
        self._size = np.array(size, dtype=np.float32)
        self._scale = scale

        self._pose = pose.copy()
        self._n_pose = pose.copy()

        self._target = target
        self._n_target = target

        self._mode = TrackballMode.ROTATE

        self._pdown = np.array([0.0, 0.0], dtype=np.float32)

    @property
    def pose(self):
        """(4, 4) float : The camera-to-world pose of the camera."""
        return self._n_pose

    @property
    def target(self):
        """(3,) float : The point that the camera orbits about."""
        return self._n_target

    def set_mode(self, mode):
        """Set the motion that dragging the mouse gives the camera.

        Parameters
        ----------
        mode : TrackballMode
            The motion of the next drags.
        """
        self._mode = mode

    def resize(self, size):
        """Resize the window.

        Parameters
        ----------
        size : (float, float)
            The new width and height of the camera image in pixels.
        """
        self._size = np.array(size)

    def down(self, point):
        """Record an initial mouse press at a given point.

        Parameters
        ----------
        point : (2,) int
            The x and y pixel coordinates of the mouse press.
        """
        self._pdown = np.array(point, dtype=np.float32)
        self._pose = self._n_pose
        self._target = self._n_target

    def drag(self, point):
        """Update the trackball during a drag.

        Parameters
        ----------
        point : (2,) int
            The current x and y pixel coordinates of the mouse during a drag.
            This will compute a movement for the trackball with the relative
            motion between this point and the one marked by down().
        """
        point = np.array(point, dtype=np.float32)
        dx, dy = point - self._pdown
        mindim = 0.3 * np.min(self._size)

        target = self._target
        x_axis = self._pose[:3, 0]
        y_axis = self._pose[:3, 1]
        z_axis = self._pose[:3, 2]
        eye = self._pose[:3, 3]

        # Interpret drag as a rotation
        if self._mode == TrackballMode.ROTATE:
            # Compute updated azimut directly. No fancy math here because this angle can controlled freely.
            roll_angle = np.arctan2(self._pose[2, 1], self._pose[2, 2])

            # safeguard for degenerate case when roll_angle = 0
            world_up_axis = np.array([0.0, 0.0, -1.0 if roll_angle < 0.0 else 1.0])
            azimuth_angle = -dx / mindim
            azimuth_transform = transformations.rotation_matrix(azimuth_angle, world_up_axis, target)

            # Compute current elevation angle
            pose_after_azimuth = azimuth_transform @ self._pose
            eye_after_azimuth = pose_after_azimuth[:3, 3]
            view_dir = target - eye_after_azimuth
            current_elevation_angle = -np.arctan2(view_dir[2], np.linalg.norm(view_dir[:2]))

            # Update elevation angle based on mouse motion
            desired_elevation_angle = current_elevation_angle - dy / mindim
            clamped_elevation_angle = np.clip(desired_elevation_angle, np.radians(-89.0), np.radians(89.0))
            delta_elevation_angle = desired_elevation_angle - current_elevation_angle

            # Compute the elevation axis
            norm_view_dir = np.linalg.norm(view_dir)
            if norm_view_dir < EPSILON:
                elevation_axis = pose_after_azimuth[:3, 0]
                delta_elevation_angle = 0.0
            else:
                view_dir_normalized = view_dir / norm_view_dir
                elevation_axis = np.cross(world_up_axis, view_dir_normalized)
                elevation_axis_norm = np.linalg.norm(elevation_axis)
                if elevation_axis_norm < EPSILON:
                    elevation_axis = pose_after_azimuth[:3, 0]
                else:
                    elevation_axis = elevation_axis / elevation_axis_norm

            # Apply the elevation rotation
            elevation_rotation = transformations.rotation_matrix(delta_elevation_angle, elevation_axis, target)
            self._n_pose = elevation_rotation @ pose_after_azimuth

            # Prevent locking the camera in the up/down direction
            self._pdown[1] -= (desired_elevation_angle - clamped_elevation_angle) * mindim

        # Interpret drag as a camera pan in view plane
        elif self._mode == TrackballMode.PAN:
            dx = -dx / (5.0 * mindim) * self._scale
            dy = -dy / (5.0 * mindim) * self._scale

            translation = dx * x_axis + dy * y_axis
            self._n_target = self._target + translation
            t_tf = np.eye(4)
            t_tf[:3, 3] = translation
            self._n_pose = t_tf @ self._pose

        # Interpret drag as a zoom motion
        elif self._mode == TrackballMode.ZOOM:
            radius = np.linalg.norm(eye - target)
            ratio = 0.0
            if dy > 0:
                ratio = np.exp(abs(dy) / (0.5 * self._size[1])) - 1.0
            elif dy < 0:
                ratio = 1.0 - np.exp(dy / (0.5 * (self._size[1])))
            translation = -np.sign(dy) * ratio * radius * z_axis
            t_tf = np.eye(4)
            t_tf[:3, 3] = translation
            self._n_pose = t_tf @ self._pose

    def scroll(self, clicks):
        """Move the camera towards the target or away from it, as the wheel of the mouse turns.

        Parameters
        ----------
        clicks : float
            The number of clicks of the wheel, positive to move towards the target.
        """
        # Each click scales the distance to the target, for the pose shown and the one a drag starts from alike
        poses = []
        for pose in (self._n_pose, self._pose):
            radius = np.linalg.norm(pose[:3, 3] - self._target)
            pose_moved = pose.copy()
            pose_moved[:3, 3] += (0.9**clicks - 1.0) * radius * pose[:3, 2]
            poses.append(pose_moved)
        self._n_pose, self._pose = poses

    def rotate(self, azimuth, axis):
        """Turn the camera about an axis through the target.

        Parameters
        ----------
        azimuth : float
            The angle of the rotation, in radians.
        axis : (3,) float
            The axis of the rotation.
        """
        rotation = transformations.rotation_matrix(azimuth, axis, self._target)
        self._n_pose = rotation @ self._n_pose
        self._pose = rotation @ self._pose

    def set_camera_pose(self, pose, target=None):
        """Place the camera, and the point it orbits about.

        Parameters
        ----------
        pose : (4, 4) float
            The camera-to-world pose.
        target : (3,) float, optional
            The point that the camera orbits about. Defaults to the point along the view axis of the camera, as far
            from it as the former target.
        """
        if target is None:
            radius = np.linalg.norm(self._n_pose[:3, 3] - self._n_target)
            target = pose[:3, 3] - radius * pose[:3, 2]
        self._pose = pose.copy()
        self._n_pose = pose.copy()
        self._target = target
        self._n_target = target
