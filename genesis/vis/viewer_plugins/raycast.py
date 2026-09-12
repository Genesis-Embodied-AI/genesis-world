from threading import Lock
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import torch
from typing_extensions import override

import genesis as gs
from genesis.ext.pyrender.camera import OrthographicCamera
from genesis.utils.misc import qd_to_numpy, qd_to_torch, with_lock
from genesis.utils.raycast import Ray, RayHit

from .base import ViewerPlugin

if TYPE_CHECKING:
    from genesis.engine.bvh import AABB, LBVH
    from genesis.engine.scene import Scene
    from genesis.engine.solvers.kinematic_solver import KinematicSolver
    from genesis.ext.pyrender.node import Node
    from genesis.utils.array_class import RaycastResult


class RaycastTarget(NamedTuple):
    """One bounding volume hierarchy (BVH) the viewer casts against, over one solver's collision or visual mesh."""

    solver: "KinematicSolver"
    aabb: "AABB"
    bvh: "LBVH"
    result: "RaycastResult"
    # (n_vfaces,) opt-in mask for a visual BVH; None for a collision BVH, which covers every face.
    vfaces_mask: torch.Tensor | None


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
        # NOTE: delayed imports to avoid pulling in rigid_solver / array_class before gs is fully initialized.
        import genesis.utils.array_class as array_class
        from genesis.engine.bvh import AABB, LBVH

        self.scene = scene
        self.envs_idx = scene._envs_idx
        self.targets: list[RaycastTarget] = []
        self._lock = Lock()

        n_envs_max = len(self.envs_idx)
        # Visual meshes exist on both solvers, while collision geometry is the rigid solver's alone.
        solvers = (scene.sim.rigid_solver, scene.sim.kinematic_solver) if use_visual_geom else (scene.sim.rigid_solver,)
        for solver in solvers:
            if not solver.is_active:
                continue
            vfaces_mask = None
            if use_visual_geom:
                vfaces_mask = solver.vfaces_raycast_mask
                if not vfaces_mask.any():
                    continue
                # The tree spans every vface, masked-out ones staying unhittable, so a leaf payload is the vface
                # index itself (see kernel_update_visual_aabbs).
                n_slots = vfaces_mask.shape[0]
            else:
                n_slots = solver.dyn_info.faces.geom_idx.shape[0]
                if n_slots == 0:
                    continue
            aabb = AABB(n_batches=n_envs_max, n_aabbs=n_slots)
            bvh = LBVH(
                aabb,
                max_n_query_result_per_aabb=0,  # Not used for ray queries
                n_radix_sort_groups=min(64, n_slots),
            )
            result = array_class.get_raycast_result(n_envs_max)
            self.targets.append(RaycastTarget(solver, aabb, bvh, result, vfaces_mask))

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
            for buffer in (target.result.hit_point, target.result.normal):
                qd_to_numpy(buffer, row_mask=0, keepdim=False, transpose=True, copy=True)

    @with_lock
    def update(self) -> None:
        """Refresh per-env vertex positions, AABBs and rebuild every BVH."""
        from genesis.utils.raycast_qd import kernel_update_verts_and_aabbs, kernel_update_visual_aabbs

        for target in self.targets:
            solver = target.solver
            if target.vfaces_mask is not None:
                # A visual vertex follows its vgeom pose, except on an entity opting into custom vverts, where it
                # follows the vverts buffer instead (see get_visual_vvert_pos). Refreshing the poses therefore covers
                # every vertex this pass owns, the buffer moving only when the user calls set_vverts.
                solver.update_forward_pos()
                solver.update_vgeoms()
                kernel_update_visual_aabbs(
                    target.vfaces_mask, solver.dyn_state, target.aabb, solver.dyn_info, solver.rigid_config
                )
            else:
                kernel_update_verts_and_aabbs(solver.dyn_state, target.aabb, solver.dyn_info, solver.rigid_config)
            target.bvh.build()

    @with_lock
    def cast(
        self, ray_origin: np.ndarray, ray_direction: np.ndarray, max_range: float = 1000.0, envs_idx=None
    ) -> RayHit | None:
        """
        Cast a single ray against the BVH of each env in parallel and return the closest hit across envs and solvers.

        `RayHit.env_idx` reports the env the hit comes from, None when the scene holds no parallel envs.

        Parameters
        ----------
        ray_origin : np.ndarray, shape (3,)
            Ray origin in world coordinates.
        ray_direction : np.ndarray, shape (3,)
            Normalized ray direction.
        max_range : float, optional
            Per-env BVH traversal max distance.
        envs_idx : optional
            Indices of envs to raycast. Defaults to all envs.
        """
        from genesis.utils.raycast_qd import kernel_cast_ray

        ray_origin = np.ascontiguousarray(ray_origin, dtype=gs.np_float)
        ray_direction = np.ascontiguousarray(ray_direction, dtype=gs.np_float)
        closest_hit: RayHit | None = None
        for target in self.targets:
            solver = target.solver
            is_visual = target.vfaces_mask is not None
            # Each pass caps its traversal at the best distance so far, so any hit it reports is closer by
            # construction and the closest hit across targets needs no further comparison.
            kernel_cast_ray(
                ray_origin,
                envs_idx if envs_idx is not None else self.envs_idx,
                target.bvh.nodes,
                target.bvh.morton_codes,
                ray_direction,
                solver.dyn_state,
                target.result,
                solver.dyn_info,
                solver.rigid_info,
                max_range if closest_hit is None else closest_hit.distance,
                eps=gs.EPS,
                is_visual=is_visual,
            )

            # A no-hit env holds +inf, so argmin lands on the closest hitting env; geom_idx then rejects the
            # all-inf case, which would otherwise report env 0 as a hit.
            distances = qd_to_torch(target.result.distance, copy=None)
            winner = int(distances.argmin())
            geom_idx = int(qd_to_numpy(target.result.geom_idx, row_mask=winner, keepdim=False))
            if geom_idx < 0:
                continue

            distance = float(distances[winner])
            # These two escape into the returned hit, which outlives the lock, while the buffers they come from are
            # rewritten by the next cast.
            position = qd_to_numpy(target.result.hit_point, row_mask=winner, keepdim=False, transpose=True, copy=True)
            normal = qd_to_numpy(target.result.normal, row_mask=winner, keepdim=False, transpose=True, copy=True)
            geoms = solver.vgeoms if is_visual else solver.geoms
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
