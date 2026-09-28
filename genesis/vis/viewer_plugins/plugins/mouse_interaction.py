from threading import Lock
from typing import TYPE_CHECKING

import numpy as np
from typing_extensions import override

import genesis as gs
import genesis.utils.geom as gu
from genesis.engine.entities.rigid_entity import RigidLink
from genesis.utils.mesh import create_cylinder, create_plane
from genesis.utils.misc import tensor_to_array, with_lock
from genesis.utils.raycast import Ray, RayHit, plane_raycast
from genesis.vis.keybindings import MouseButton

from ..base import EVENT_HANDLE_STATE, EVENT_HANDLED
from ..raycast import RaycasterViewerPlugin

if TYPE_CHECKING:
    from genesis.engine.scene import Scene
    from genesis.ext.pyrender.node import Node


MIN_PICKABLE_MASS = 1e-3  # kg - links below this threshold are skipped to avoid numerical instability
GRAB_ACC = 10.0  # m/s^2 - acceleration a grab pulls with at one 'spring_slack' from the cursor


class MouseInteractionPlugin(RaycasterViewerPlugin):
    """
    Drag rigid entities around with the mouse, by a spring attached where the cursor grabbed them.

    Parameters
    ----------
    use_force : bool, optional
        Drag the grabbed link with a spring force, leaving the solver to resolve it against gravity, contacts and the
        rest of the kinematic tree, so the drag never breaks the physics. False instead moves the link onto the cursor
        outright, which tracks exactly but drives it through anything in the way and jolts whatever it is attached to.
        Default is True.
    spring_slack : float, optional
        How far below the cursor a grabbed link hangs once it settles, in meters. The stiffness of the spring follows
        from this and the mass of the link, so heavy and light entities are equally responsive. Lower it for a grab
        that tracks the cursor closely, at the cost of pushing entities through whatever they touch on the way; raise
        it to let contacts win instead, at the cost of trailing behind a fast cursor. The distance is quoted at
        standard gravity; a weaker-gravity scene holds its links closer still. Only used when `use_force` is True.
        Default is 0.02.
    color : tuple of float, optional
        RGBA color of the line, handle and drag plane drawn while an entity is held. Default is (0.2, 0.8, 0.8, 0.6).
    use_visual_geom : bool, optional
        See `RaycasterViewerPlugin`. Default is False.
    """

    def __init__(
        self,
        use_force: bool = True,
        spring_slack: float = 0.02,
        color: tuple[float, float, float, float] = (0.2, 0.8, 0.8, 0.6),
        use_visual_geom: bool = False,
    ) -> None:
        super().__init__(use_visual_geom)
        self.use_force = bool(use_force)
        self.spring_slack = float(spring_slack)
        self.color = tuple(color)

        self._lock: Lock = Lock()
        self._held_link: RigidLink | None = None
        # The plugin is active only while the simulation advances. _last_step_t is the step counter seen at the
        # previous sim-step callback, and _sim_running caches whether it advanced, so that on_draw and the mouse
        # handlers, which run off the step loop, share the same paused or running verdict.
        self._last_step_t: int = 0
        self._sim_running: bool = False
        self._interact_env_idx: int | None = None
        self._env_offset: np.ndarray = np.zeros(3, dtype=gs.np_float)
        self._held_point_local: np.ndarray | None = None
        self._mouse_drag_plane: tuple[np.ndarray, float] | None = None
        self._prev_mouse_screen_pos: tuple[int, int] = (0, 0)
        self._prev_mouse_scene_pos: np.ndarray | None = None
        self._surface_normal: np.ndarray | None = None
        self._plane_rotation_angle: float = 0.0

        # Persistent debug nodes (lifecycle managed in on_draw)
        self._debug_interact_nodes: list["Node"] = []
        self._debug_normal_node: "Node | None" = None
        self._unit_cylinder_mesh = None
        self._plane_mesh = None

    def build(self, viewer, camera: "Node", scene: "Scene"):
        super().build(viewer, camera, scene)
        self._last_step_t = self.scene.sim.cur_step_global
        self._prev_mouse_screen_pos = (self.viewer._viewport_size[0] // 2, self.viewer._viewport_size[1] // 2)

        self._unit_cylinder_mesh = create_cylinder(radius=0.005, height=1.0, color=self.color)
        plane_color = (*self.color[:3], self.color[3] * 0.5)
        plane_vmesh, _ = create_plane(plane_size=(0.3, 0.3), color_or_texture=plane_color, double_sided=True)
        self._plane_mesh = plane_vmesh

    @override
    def on_mouse_motion(self, x: int, y: int, dx: int, dy: int) -> EVENT_HANDLE_STATE:
        self._prev_mouse_screen_pos = (x, y)

    @with_lock
    @override
    def on_mouse_drag(self, x: int, y: int, dx: int, dy: int, buttons: int, modifiers: int) -> EVENT_HANDLE_STATE:
        self._prev_mouse_screen_pos = (x, y)
        if self._held_link:
            return EVENT_HANDLED

    @with_lock
    @override
    def on_mouse_scroll(self, x: int, y: int, scroll_x: float, scroll_y: float) -> EVENT_HANDLE_STATE:
        if self._held_link and self._surface_normal is not None:
            # Rotate the drag plane around the surface normal
            self._plane_rotation_angle += scroll_y * 0.1  # 0.1 radians per scroll unit
            self._update_drag_plane()
            return EVENT_HANDLED

    @with_lock
    @override
    def on_mouse_press(self, x: int, y: int, button: int, modifiers: int) -> EVENT_HANDLE_STATE:
        if not self._sim_running:  # no grabbing while the simulation is paused
            return
        if button == MouseButton.LEFT:  # left mouse button
            ray = self._screen_position_to_ray(x, y)
            ray_hit = self._raycaster.cast(ray[0], ray[1])
            if ray_hit is None:
                return

            # Dragging applies forces, so only a movable rigid link qualifies: a visual cast also reaches the
            # kinematic solver, whose links carry no mass.
            if ray_hit.geom is not None and isinstance(ray_hit.geom.link, RigidLink) and not ray_hit.geom.link.is_fixed:
                link = ray_hit.geom.link
                hit_env_idx = ray_hit.env_idx
                # The mass a link was built with, which a setter may only ever bring to another valid mass, so the
                # affordance costs no reading of the solver from the thread that draws it.
                mass = link.desc.mass

                # Validate mass is not too small to prevent numerical instability
                if mass < MIN_PICKABLE_MASS:
                    gs.logger.warning(
                        f"Link '{link.name}' has very small mass ({mass:.2e}). "
                        "Skipping interaction to avoid numerical instability."
                    )
                    return

                self._held_link = link
                self._interact_env_idx = hit_env_idx
                self._env_offset = (
                    self.scene.envs_offset[hit_env_idx].astype(gs.np_float, copy=False)
                    if hit_env_idx is not None
                    else np.zeros(3, dtype=gs.np_float)
                )

                # Store the surface normal for rotation
                self._surface_normal = ray_hit.normal
                self._plane_rotation_angle = 0.0
                self._prev_mouse_scene_pos = ray_hit.position

                # Create drag plane perpendicular to surface normal
                self._update_drag_plane()

                # Store held point in link-local frame. ray_hit.position is in world coords (with envs_offset baked in
                # by the raycaster), while link.get_pos returns env-local coords, so subtract the offset to align.
                link_pos = tensor_to_array(link.get_pos(envs_idx=hit_env_idx, relative=False).squeeze(0))
                link_quat = tensor_to_array(link.get_quat(envs_idx=hit_env_idx, relative=False).squeeze(0))
                hit_env_local = ray_hit.position - self._env_offset
                self._held_point_local = gu.inv_transform_by_trans_quat(hit_env_local, link_pos, link_quat)

    @with_lock
    @override
    def on_mouse_release(self, x: int, y: int, button: int, modifiers: int) -> EVENT_HANDLE_STATE:
        if button == MouseButton.LEFT:
            self._held_link = None
            self._interact_env_idx = None
            self._env_offset = np.zeros(3, dtype=gs.np_float)
            self._held_point_local = None
            self._mouse_drag_plane = None
            self._prev_mouse_scene_pos = None
            self._surface_normal = None
            self._plane_rotation_angle = 0.0

    @with_lock
    @override
    def update_on_sim_step(self) -> None:
        # Active only while the simulation advances: a grabbed body is dragged through physics, so a step that did not
        # advance the counter (paused) releases the grip and applies no force or motion. on_draw reads _sim_running to
        # drop its hover/drag visuals too.
        self._sim_running = self.scene.sim.cur_step_global > self._last_step_t
        self._last_step_t = self.scene.sim.cur_step_global
        if not self._sim_running:
            self._held_link = None
            return

        super().update_on_sim_step()

        if self._held_link:
            mouse_ray: Ray = self._screen_position_to_ray(*self._prev_mouse_screen_pos)
            assert self._mouse_drag_plane is not None
            ray_hit_plane: RayHit | None = plane_raycast(*self._mouse_drag_plane, mouse_ray)

            # If ray doesn't hit the plane, skip this update
            if ray_hit_plane is None:
                return

            self._prev_mouse_scene_pos = ray_hit_plane.position
            envs_idx = self._interact_env_idx

            if self.use_force:
                self._apply_spring_force(ray_hit_plane.position, self.scene.sim.dt)
            else:
                assert self._held_point_local is not None
                link_quat = tensor_to_array(self._held_link.get_quat(envs_idx=envs_idx, relative=False).squeeze(0))
                offset_world = gu.transform_by_quat(self._held_point_local, link_quat)
                # Mouse target is world; entity position is env-local, so strip the env offset before setting.
                new_link_pos = ray_hit_plane.position - self._env_offset - offset_world
                self._held_link.entity.set_pos(new_link_pos, envs_idx=envs_idx, relative=False)

    @with_lock
    @override
    def on_draw(self) -> None:
        if self.scene._visualizer is None or not self.scene._visualizer.is_built:
            return
        # When detached at runtime, copy-on-write removal already dropped this plugin from viewer.plugins, but an
        # in-flight dispatch loop may still walk its pre-removal snapshot and call us once more. Skip drawing so we
        # do not re-create the debug nodes on_close just cleared.
        if self not in self.viewer.plugins:
            return

        mouse_ray: Ray = self._screen_position_to_ray(*self._prev_mouse_screen_pos)

        if self._held_link:
            # Clean up hover arrow when transitioning to hold state
            if self._debug_normal_node is not None:
                self.scene.clear_debug_object(self._debug_normal_node)
                self._debug_normal_node = None

            assert self._mouse_drag_plane is not None
            assert self._held_point_local is not None

            envs_idx = self._interact_env_idx
            link_pos = tensor_to_array(self._held_link.get_pos(envs_idx=envs_idx, relative=False).squeeze(0))
            link_quat = tensor_to_array(self._held_link.get_quat(envs_idx=envs_idx, relative=False).squeeze(0))
            held_point_env_local = gu.transform_by_trans_quat(self._held_point_local, link_pos, link_quat)
            held_point_world = held_point_env_local + self._env_offset

            plane_hit: RayHit | None = plane_raycast(*self._mouse_drag_plane, mouse_ray)
            if plane_hit is not None:
                control_point = plane_hit.position

                # Sphere at clamped control point (translation only, no rotation)
                sphere_T = gu.trans_to_T(control_point)
                # Cylinder from held point to clamped control point
                line_T = self._compute_line_T(held_point_world, control_point)
                # Drag plane visualization centered on clamped control point
                plane_T = gu.trans_R_to_T(control_point, gu.z_up_to_R(self._mouse_drag_plane[0]))

                if not self._debug_interact_nodes:
                    self._debug_interact_nodes.append(
                        self.scene.draw_debug_sphere(control_point, radius=0.01, color=self.color)
                    )
                    self._debug_interact_nodes.append(self.scene.draw_debug_mesh(self._unit_cylinder_mesh, T=line_T))
                    self._debug_interact_nodes.append(self.scene.draw_debug_mesh(self._plane_mesh, T=plane_T))
                else:
                    self.scene.update_debug_objects(self._debug_interact_nodes, (sphere_T, line_T, plane_T))
            else:
                # No plane hit: hide held visualization nodes
                for node in self._debug_interact_nodes:
                    self.scene.clear_debug_object(node)
                self._debug_interact_nodes.clear()

        else:
            # Clean up held visualization nodes
            for node in self._debug_interact_nodes:
                self.scene.clear_debug_object(node)
            self._debug_interact_nodes.clear()

            # Hover arrow: only show for pickable (non-fixed, sufficient mass) entities
            closest_hit: RayHit | None = self._raycaster.cast(mouse_ray[0], mouse_ray[1])
            link = closest_hit.geom.link if closest_hit is not None and closest_hit.geom is not None else None
            is_pickable = (
                self._sim_running
                and isinstance(link, RigidLink)
                and not link.is_fixed
                and link.desc.mass >= MIN_PICKABLE_MASS
            )
            if is_pickable:
                arrow_T = gu.trans_R_to_T(closest_hit.position, gu.z_up_to_R(closest_hit.normal))
                if self._debug_normal_node is None:
                    self._debug_normal_node = self.scene.draw_debug_arrow(
                        closest_hit.position, closest_hit.normal * 0.25, color=self.color
                    )
                else:
                    self.scene.update_debug_objects((self._debug_normal_node,), (arrow_T,))
            elif self._debug_normal_node is not None:
                self.scene.clear_debug_object(self._debug_normal_node)
                self._debug_normal_node = None

    @with_lock
    @override
    def on_close(self) -> None:
        # Release any held link and drop the debug visuals so detaching the plugin (or viewer teardown) leaves no
        # orphaned hover arrow or drag-plane nodes behind.
        self._held_link = None
        if self.scene._visualizer is not None and self.scene._visualizer.is_built:
            for node in self._debug_interact_nodes:
                self.scene.clear_debug_object(node)
            if self._debug_normal_node is not None:
                self.scene.clear_debug_object(self._debug_normal_node)
        self._debug_interact_nodes.clear()
        self._debug_normal_node = None

    def _compute_line_T(self, start: np.ndarray, end: np.ndarray) -> np.ndarray:
        """Compute transform for unit cylinder (height=1, centered at z=0) from start to end."""
        direction = end - start
        length = float(np.linalg.norm(direction))
        if length < gs.EPS:
            return gu.trans_to_T(start)
        R_basis = gu.z_up_to_R(direction / length)
        T = np.eye(4, dtype=gs.np_float)
        T[:3, 0] = R_basis[:, 0]
        T[:3, 1] = R_basis[:, 1]
        T[:3, 2] = direction  # scaled z: maps local +/-0.5 to world start/end
        T[:3, 3] = (start + end) * 0.5
        return T

    def _update_drag_plane(self) -> None:
        """Update the drag plane based on surface normal and rotation angle."""
        if self._surface_normal is None or self._prev_mouse_scene_pos is None:
            return

        # Get camera direction
        cam_forward = np.ascontiguousarray(-self.camera.matrix[:3, 2], dtype=gs.np_float)
        surface_normal_contig = np.ascontiguousarray(self._surface_normal, dtype=gs.np_float)

        # Create orthonormal basis with surface_normal as z-axis
        R = gu.z_up_to_R(surface_normal_contig, up=cam_forward)

        plane_normal = R[:, 0] * np.dot(R[:, 0], cam_forward) + R[:, 1] * np.dot(R[:, 1], cam_forward)
        plane_normal = plane_normal / (np.linalg.norm(plane_normal) + gs.EPS)

        if abs(self._plane_rotation_angle) > gs.EPS:
            rotation_matrix = gu.axis_angle_to_R(surface_normal_contig, self._plane_rotation_angle)
            plane_normal = gu.transform_by_R(plane_normal, rotation_matrix)

        # Set the drag plane (perpendicular to surface normal)
        self._mouse_drag_plane = (plane_normal, -np.dot(plane_normal, self._prev_mouse_scene_pos))

    def _apply_spring_force(self, control_point: np.ndarray, dt: float) -> None:
        if not self._held_link:
            return
        envs_idx = self._interact_env_idx

        # Current link state in env-local frame
        link_pos = tensor_to_array(self._held_link.get_pos(envs_idx=envs_idx, relative=False).squeeze(0))
        link_quat = tensor_to_array(self._held_link.get_quat(envs_idx=envs_idx, relative=False).squeeze(0))
        lin_vel = tensor_to_array(self._held_link.get_vel(envs_idx=envs_idx, relative=False).squeeze(0))
        ang_vel = tensor_to_array(self._held_link.get_ang(envs_idx=envs_idx).squeeze(0))

        # Held point in env-local frame; control point comes from a world-space plane raycast, so strip the offset.
        held_point_env_local = gu.transform_by_trans_quat(self._held_point_local, link_pos, link_quat)
        control_point_env_local = control_point - self._env_offset

        # What the dynamics spins this link about, read live: a center of mass or an inertia set at runtime moves it,
        # and a drag worked out around the authored one pulls about a point the solver does not turn on. Per
        # environment only where link info is batched, which is the only case where the grabbed one has its own.
        solver = self._held_link.solver
        envs_idx_info = self._interact_env_idx if solver.is_links_info_batched else None
        com_local = tensor_to_array(solver.get_links_COM(self._held_link.idx, envs_idx_info))[..., 0, :]
        inertia_local = tensor_to_array(solver.get_links_inertia(self._held_link.idx, envs_idx_info))[..., 0, :, :]
        if envs_idx_info is not None:
            com_local, inertia_local = com_local[0], inertia_local[0]
        inertial_quat = tensor_to_array(self._held_link.desc.inertial_quat)
        world_principal_quat = gu.transform_quat_by_quat(inertial_quat, link_quat)

        # Compute arm from COM to held point in world frame
        arm_in_principal = gu.inv_transform_by_trans_quat(self._held_point_local, com_local, inertial_quat)
        arm_in_world = gu.transform_by_quat(arm_in_principal, world_principal_quat)

        # Compute inverse inertia in world frame
        R_world = gu.quat_to_R(world_principal_quat)
        inertia_world = R_world @ inertia_local @ R_world.T
        inv_inertia_world = np.linalg.inv(inertia_world)

        pos_err_v = control_point_env_local - held_point_env_local
        # The mass of the environment being grabbed in, where a batched scene gives each one its own.
        mass = float(self._held_link.get_mass(envs_idx=envs_idx_info))
        inv_mass = 1.0 / mass

        # A link at rest hangs below its center of mass with the spring carrying its whole weight, so the slack alone
        # fixes the natural frequency. The correction offsets the step of free fall the softened solve settles short.
        nominal_freq = np.sqrt(GRAB_ACC / self.spring_slack)
        natural_freq = nominal_freq / max(1.0 - dt * nominal_freq, gs.EPS)
        spring_const = mass * natural_freq**2

        total_impulse = np.zeros(3, dtype=gs.np_float)

        # Approximate spring-damper in each axis
        for i in range(3):
            body_point_vel = lin_vel + np.cross(ang_vel, arm_in_world)
            vel_err_v = -body_point_vel

            direction = np.zeros(3, dtype=gs.np_float)
            direction[i] = 1.0

            pos_err = np.dot(direction, pos_err_v)
            vel_err = np.dot(direction, vel_err_v)

            # Compute virtual mass (effective inertia for this constraint direction)
            arm_x_dir = np.cross(arm_in_world, direction)
            rot_mass = np.dot(arm_x_dir, inv_inertia_world @ arm_x_dir)
            virtual_mass = 1.0 / (inv_mass + rot_mass + gs.EPS)

            # Critical damping
            damping_coeff = 2.0 * np.sqrt(spring_const * virtual_mass)
            # Solving for the impulse that lands the end-of-step velocity on the spring-damper response holds that
            # damping ratio at any stiffness; a held force would only do so while dt stays below 1 / frequency.
            softness = 1.0 / (dt * (damping_coeff + dt * spring_const))
            bias_rate = spring_const / (damping_coeff + dt * spring_const)
            impulse = (vel_err + bias_rate * pos_err) / (inv_mass + rot_mass + softness)

            lin_vel += direction * impulse * inv_mass
            ang_vel += inv_inertia_world @ (arm_x_dir * impulse)

            total_impulse[i] = impulse

        # A rotation about the grabbed point leaves that point where it is, so the spring-damper above is blind to the
        # pendulum swinging under the cursor. Twice its frequency damps it critically at worst, never under, and the
        # cap keeps a grab near the center of mass, where the pendulum degenerates, from freezing rotation outright.
        arm_length = np.linalg.norm(arm_in_world)
        swing_freq = min(2.0 * np.sqrt(GRAB_ACC / (arm_length + gs.EPS)), natural_freq)
        # That swing pivots about the grabbed point, hence the inertia about it, and solving for the end-of-step
        # angular velocity keeps the decay from overshooting when the two inertias differ widely.
        swing_inertia = inertia_world + mass * (arm_length**2 * np.eye(3) - np.outer(arm_in_world, arm_in_world))
        ang_vel_next = np.linalg.solve(inertia_world + dt * swing_freq * swing_inertia, inertia_world @ ang_vel)
        torque = -swing_freq * (swing_inertia @ ang_vel_next)

        # The force acts at the grabbed point, so its moment about the center of mass swings the body to the cursor.
        self._held_link.apply_external_wrench(total_impulse / dt, torque, envs_idx=envs_idx, pos=held_point_env_local)
