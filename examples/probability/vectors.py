"""Measured rigid-body diagnostics and display-only geometry, all in world coordinates."""

from dataclasses import dataclass

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation


FLOOR_COLOR = (0.55, 0.72, 0.86, 0.7)
LINEAR_COLOR = (0.05, 0.9, 1.0, 1.0)
ANGULAR_COLOR = (0.95, 0.3, 0.95, 1.0)
INITIAL_LINEAR_COLOR = (0.25, 0.65, 0.9, 0.7)
INITIAL_ANGULAR_COLOR = (0.65, 0.35, 0.95, 0.7)
NORMAL_COLOR = (0.25, 0.55, 1.0, 1.0)
STICK_COLOR = (0.2, 0.95, 0.35, 1.0)
SLIP_COLOR = (1.0, 0.35, 0.08, 1.0)
CONTACT_COLOR = (1.0, 0.9, 0.2, 1.0)
GHOST_COLOR = (1.0, 0.78, 0.22, 0.23)
TRAIL_COLOR = (0.9, 0.78, 0.4, 0.65)
SLIP_THRESHOLD = 0.005  # m/s; a visualization diagnostic, not an engine friction parameter


@dataclass(frozen=True)
class Layers:
    floor_normals: bool = True
    initial_pose: bool = True
    initial_linear: bool = True
    initial_angular: bool = True
    linear: bool = True
    angular: bool = True
    contact_points: bool = True
    normal_forces: bool = True
    friction_forces: bool = True
    trail: bool = True
    velocity_scale: float = 0.12  # metres drawn per m/s
    angular_scale: float = 0.012  # metres drawn per rad/s
    force_scale: float = 1.0  # metres drawn per newton
    arrow_limit: float = 0.45  # metres; capped arrows are identified in the UI


@dataclass
class Contact:
    position: np.ndarray
    normal: np.ndarray
    force: np.ndarray  # total solver force ON the coin/die, in newtons
    normal_force: np.ndarray
    friction_force: np.ndarray
    slip_velocity: np.ndarray
    regime: str


@dataclass
class Kinematics:
    center: np.ndarray
    quaternion: np.ndarray
    linear: np.ndarray
    angular: np.ndarray


@dataclass
class Diagnostics:
    center: np.ndarray
    quaternion: np.ndarray
    linear: np.ndarray
    angular: np.ndarray
    initial_center: np.ndarray
    initial_quaternion: np.ndarray
    initial_linear: np.ndarray
    initial_angular: np.ndarray
    floor_origin: np.ndarray
    floor_normal: np.ndarray
    floor_friction: float
    entity_friction: float
    friction: float
    contacts: list[Contact]
    contact_time: float
    trail: np.ndarray


def contact_components(position, normal, force, center, linear, angular, floor_velocity) -> Contact:
    """Decompose a measured force, and measure point slip using rigid-body kinematics."""
    normal = normal / np.linalg.norm(normal)
    normal_force = np.dot(force, normal) * normal
    friction_force = force - normal_force
    point_velocity = linear + np.cross(angular, position - center) - floor_velocity
    slip_velocity = point_velocity - np.dot(point_velocity, normal) * normal
    if np.linalg.norm(normal_force) < 1e-8:
        regime = "unloaded"
    else:
        regime = "slipping" if np.linalg.norm(slip_velocity) > SLIP_THRESHOLD else "sticking"
    return Contact(position, normal, force, normal_force, friction_force, slip_velocity, regime)


def display_vector(vector: np.ndarray, scale: float, limit: float) -> tuple[np.ndarray, bool]:
    """Apply a labelled linear scale; keep the measured data untouched when clipping the display."""
    scaled = vector * scale
    length = np.linalg.norm(scaled)
    capped = length > limit
    return scaled * (limit / length) if capped else scaled, bool(capped)


class VectorRenderer:
    def __init__(self, scene, geom):
        self.scene = scene
        self.ghost = geom.mesh.trimesh.copy()
        transform = np.eye(4)
        transform[:3, :3] = Rotation.from_quat(geom.init_quat, scalar_first=True).as_matrix()
        transform[:3, 3] = geom.init_pos
        self.ghost.apply_transform(transform)
        self.ghost.visual = trimesh.visual.ColorVisuals(self.ghost, face_colors=np.array(GHOST_COLOR))

    def arrow(self, origin, vector, scale, color, layers: Layers):
        scaled, _ = display_vector(vector, scale, layers.arrow_limit)
        length = np.linalg.norm(scaled)
        if length > 1e-6:
            self.scene.draw_debug_arrow(origin, scaled, radius=min(0.002, 0.05 * length), color=color)

    def draw(self, data: Diagnostics, layers: Layers):
        # These are debug meshes, never entities: they cannot change mass, contacts or trajectories.
        self.scene.clear_debug_objects()
        if layers.floor_normals:
            normal = data.floor_normal
            axis = np.eye(3)[np.argmin(np.abs(normal))]
            tangent = np.cross(normal, axis)
            tangent /= np.linalg.norm(tangent)
            bitangent = np.cross(normal, tangent)
            projected = data.center - np.dot(data.center - data.floor_origin, normal) * normal
            for u in (-0.22, 0.0, 0.22):
                for v in (-0.22, 0.0, 0.22):
                    self.arrow(projected + u * tangent + v * bitangent, normal, 0.08, FLOOR_COLOR, layers)
        if layers.initial_pose:
            transform = np.eye(4)
            transform[:3, :3] = Rotation.from_quat(data.initial_quaternion, scalar_first=True).as_matrix()
            transform[:3, 3] = data.initial_center
            self.scene.draw_debug_mesh(self.ghost, T=transform)
            self.scene.draw_debug_frame(transform, axis_length=0.09, origin_size=0.005, axis_radius=0.0015)
            self.scene.draw_debug_sphere(data.initial_center, radius=0.005, color=GHOST_COLOR[:3] + (1.,))
        if layers.initial_linear:
            self.arrow(data.initial_center, data.initial_linear, layers.velocity_scale, INITIAL_LINEAR_COLOR, layers)
        if layers.initial_angular:
            self.arrow(data.initial_center, data.initial_angular, layers.angular_scale, INITIAL_ANGULAR_COLOR, layers)
        if layers.linear:
            self.arrow(data.center, data.linear, layers.velocity_scale, LINEAR_COLOR, layers)
        if layers.angular:
            self.arrow(data.center, data.angular, layers.angular_scale, ANGULAR_COLOR, layers)
        for contact in data.contacts:
            if layers.contact_points:
                self.scene.draw_debug_sphere(contact.position, radius=0.004, color=CONTACT_COLOR)
            if layers.normal_forces:
                self.arrow(contact.position, contact.normal_force, layers.force_scale, NORMAL_COLOR, layers)
            if layers.friction_forces:
                color = SLIP_COLOR if contact.regime == "slipping" else STICK_COLOR
                if contact.regime == "unloaded":
                    color = FLOOR_COLOR
                self.arrow(contact.position, contact.friction_force, layers.force_scale, color, layers)
        if layers.trail and len(data.trail) > 1:
            # Bound the number of display segments; every point still comes from the simulated path.
            indices = np.unique(np.linspace(0, len(data.trail) - 1, min(60, len(data.trail))).astype(int))
            points = data.trail[indices]
            for start, end in zip(points[:-1], points[1:]):
                if np.linalg.norm(end - start) > 1e-6:
                    self.scene.draw_debug_line(start, end, radius=0.001, color=TRAIL_COLOR)
