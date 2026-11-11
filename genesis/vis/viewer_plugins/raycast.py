from threading import Lock
from typing import TYPE_CHECKING, override

import numpy as np
import torch

import genesis as gs
from genesis.ext.pyrender.camera import OrthographicCamera
from genesis.utils.misc import qd_to_numpy, qd_to_torch, with_lock
from genesis.utils.raycast import Ray, RayHit
from genesis.utils.raycast_qd import RaycastResult, RaycastTarget, get_raycast_targets, kernel_cast_ray

from .base import ViewerPlugin

if TYPE_CHECKING:
    from genesis.engine.scene import Scene
    from genesis.ext.pyrender.node import Node


class Raycaster:
    """
    Bounding volume hierarchy (BVH) accelerated single-ray cast for the viewer.

    The per-env raycast (`kernel_cast_ray`) writes one hit per env into a batched `RaycastResult`; this class then
    reduces across envs in torch to pick the closest hit. Cross-env reduction is intentionally a viewer-side concern,
    not part of the kernel, because parallel envs are otherwise meant to be isolated.

    Casts are serialized and each outcome travels entirely in the returned `RayHit`: all callers share the result
    buffers the kernel writes into, so the viewer thread picking under the cursor and the stepping thread casting its
    own rays would otherwise read each other's hit.

    Parameters
    ----------
    scene : Scene
        Scene whose geometry the rays are cast against.
    use_visual_geom : bool, optional
        Cast against the visual meshes of both the rigid and the kinematic solver, restricted to the entities opting
        in through `material.use_visual_raycasting` - the same opt-in the raycaster sensors use, so the two describe
        the same pickable geometry. `RayHit.geom` is then a `RigidVisGeom`. See `RaycasterViewerPlugin` for the
        tradeoff this carries for the user. Default is False.
    """

    def __init__(self, scene: "Scene", use_visual_geom: bool = False):
        self.scene = scene
        self._envs_offset = torch.as_tensor(scene.envs_offset, dtype=gs.tc_float, device=gs.device)
        self.targets: list[RaycastTarget] = []
        self._lock = Lock()

        n_envs_max = scene._B
        # Visual meshes exist on both solvers, while collision geometry is the rigid solver's alone.
        solvers = (scene.sim.rigid_solver, scene.sim.kinematic_solver) if use_visual_geom else (scene.sim.rigid_solver,)
        for solver in solvers:
            if solver.is_active:
                # The targets of a solver record their hits in one result, which the viewer reads after each of them
                hits = RaycastResult(sizes=dict(envs=n_envs_max))
                self.targets.extend(get_raycast_targets(solver, use_visual_geom, hits))

        if not self.targets:
            what = (
                "visual mesh opting in through material.use_visual_raycasting" if use_visual_geom else "collision face"
            )
            gs.logger.warning(f"Scene holds no {what}, viewer raycasting will not work.")
            return

        self.update()

        # FIXME: quadrants#887 - a host-side read is refused while another thread traces a kernel, so the first
        # interactive cast races the compilation of any kernel. Every buffer it reads is therefore converted up front.
        self.cast(ray_origin=np.zeros(3, dtype=gs.np_float), ray_direction=np.zeros(3, dtype=gs.np_float))
        for target in self.targets:
            for buffer in (target.hits.hit_point, target.hits.normal):
                qd_to_numpy(buffer, row_mask=0, keepdim=False, transpose=True, copy=True)

    @with_lock
    def update(self) -> None:
        """Fit every target to the current poses of its geoms."""
        for target in self.targets:
            target.update()

    @with_lock
    def cast(self, ray_origin, ray_direction, max_range: float = 1000.0, envs_idx=None) -> RayHit | None:
        """
        Cast a single ray against the BVH of each env in parallel and return the closest hit across envs and solvers.

        The ray and the hit are in the displayed world, where each env sits at its offset (see Scene.envs_offset):
        the physics of every env is in its own coordinates, so the ray is shifted into each env and the hit back out.
        `RayHit.env_idx` reports the env the hit comes from, None when the scene holds no parallel envs.

        Parameters
        ----------
        ray_origin : array-like, shape (3,)
            Ray origin in the displayed world.
        ray_direction : array-like, shape (3,)
            Normalized ray direction.
        max_range : float, optional
            Per-env BVH traversal max distance.
        envs_idx : optional
            Indices of envs to raycast. Defaults to all envs.
        """
        ray_origin = torch.as_tensor(ray_origin, dtype=gs.tc_float, device=gs.device)
        ray_direction = torch.as_tensor(ray_direction, dtype=gs.tc_float, device=gs.device)
        rays_origin = ray_origin - self._envs_offset
        envs_idx = self.scene._sanitize_envs_idx(envs_idx)
        closest_hit: RayHit | None = None
        for target in self.targets:
            # Each pass caps its traversal at the best distance so far, so any hit it reports is closer by
            # construction and the closest hit across targets needs no further comparison.
            kernel_cast_ray(
                envs_idx,
                max_range=max_range if closest_hit is None else closest_hit.distance,
                rays_start=rays_origin,
                ray_direction=ray_direction,
                target=target,
            )

            # A no-hit env holds +inf, so argmin lands on the closest hitting env; geom_idx then rejects the
            # all-inf case, which would otherwise report env 0 as a hit.
            distances = qd_to_torch(target.hits.distance, copy=None)
            winner = int(distances.argmin())
            geom_idx = int(qd_to_numpy(target.hits.geom_idx, row_mask=winner, keepdim=False))
            if geom_idx < 0:
                continue

            distance = float(distances[winner])
            position = qd_to_numpy(target.hits.hit_point, row_mask=winner, keepdim=False, transpose=True)
            position = position + self.scene.envs_offset[winner]
            # The normal escapes into the returned hit, which outlives the lock, while the next cast rewrites its buffer
            normal = qd_to_numpy(target.hits.normal, row_mask=winner, keepdim=False, transpose=True, copy=True)
            geoms = target.solver.vgeoms if target.is_visual else target.solver.geoms
            geom = geoms[geom_idx] if 0 <= geom_idx < len(geoms) else None

            closest_hit = RayHit(distance, position, normal, geom, winner if self.scene.n_envs > 0 else None)
        return closest_hit


class RaycasterViewerPlugin(ViewerPlugin):
    """
    Base viewer plugins using mouse raycast

    Parameters
    ----------
    use_visual_geom : bool, optional
        Cast against the visual meshes rather than the collision meshes, so picking follows what is drawn on screen and
        reaches entities carrying visual geometry alone. The cost is re-scanning every visual triangle on each step,
        several times the price of the collision meshes on detailed geometry, and the geometry reported back is the
        visual one, which the physics ignores. Only entities whose material sets use_visual_raycasting=True are visible
        to the cast. Default is False.
    """

    def __init__(self, use_visual_geom: bool = False) -> None:
        super().__init__()
        self.use_visual_geom = use_visual_geom
        self._raycaster: "Raycaster | None" = None

    def build(self, viewer, camera: "Node", scene: "Scene"):
        super().build(viewer, camera, scene)

        self._raycaster = Raycaster(self.scene, self.use_visual_geom)

    @override
    def update_on_sim_step(self) -> None:
        super().update_on_sim_step()

        self._raycaster.update()

    def _screen_position_to_ray(self, x: float, y: float) -> Ray:
        """
        Converts 2D screen position to a ray.

        Parameters
        ----------
        x : float
            The x coordinate on the screen.
        y : float
            The y coordinate on the screen.

        Returns
        -------
        origin : np.ndarray, shape (3,)
            The origin of the ray in world coordinates.
        direction : np.ndarray, shape (3,)
            The direction of the ray in world coordinates.
        """

        viewport_size = self.viewer._viewport_size
        w_raw = float(viewport_size[0])
        h_raw = float(viewport_size[1])
        h = max(h_raw, 1e-8)
        x_c = float(x) - 0.5 * w_raw
        y_c = float(y) - 0.5 * h_raw
        sx = 2.0 * x_c / h
        sy = 2.0 * y_c / h

        # NOTE: ignoring pixel aspect ratio; projection may change after build (e.g. O key)
        mtx = self.camera.matrix
        position = mtx[:3, 3]
        forward = -mtx[:3, 2]
        right = mtx[:3, 0]
        up = mtx[:3, 1]

        cam = self.camera.camera
        if isinstance(cam, OrthographicCamera):
            ymag = float(cam.ymag)
            origin = position + right * (sx * ymag) + up * (sy * ymag)
            direction = forward / np.linalg.norm(forward)
            return Ray(origin, direction)

        tan_half = float(np.tan(0.5 * float(cam.yfov)))
        direction = forward + right * (sx * tan_half) + up * (sy * tan_half)
        direction /= np.linalg.norm(direction)
        return Ray(position, direction)
