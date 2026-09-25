from typing import TYPE_CHECKING, NamedTuple

import numpy as np

import genesis as gs

if TYPE_CHECKING:
    from genesis.engine.entities.rigid_entity.rigid_geom import RigidGeom, RigidVisGeom


class Ray(NamedTuple):
    origin: np.ndarray  # (3,)
    direction: np.ndarray  # (3,)


class RayHit(NamedTuple):
    distance: float
    position: np.ndarray  # (3,)
    normal: np.ndarray  # (3,)
    geom: "RigidGeom | RigidVisGeom | None"
    # Env holding the geometry that was hit, None for a cast that is not per-env, e.g. against an analytical plane.
    env_idx: int | None = None


def plane_raycast(normal: np.ndarray, distance: float, ray: Ray) -> RayHit | None:
    assert normal.shape == ray.direction.shape == ray.origin.shape == (3,)
    dot = np.dot(ray.direction, normal)

    # Ray is parallel to plane
    if abs(dot) < gs.EPS:
        return None

    # Compute distance along ray to plane
    dist_along_ray = -(np.dot(ray.origin, normal) + distance) / dot

    # Intersection is behind the ray origin
    if dist_along_ray < 0:
        return None

    hit_pos = ray.origin + ray.direction * dist_along_ray
    return RayHit(distance=dist_along_ray, position=hit_pos, normal=normal, geom=None)
