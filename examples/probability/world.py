"""Genesis rigid bodies with decorative, non-colliding, massless markings."""

import xml.etree.ElementTree as ET
from collections.abc import Callable, Generator

import genesis as gs
import numpy as np
from scipy.spatial.transform import Rotation

from genesis.utils.misc import tensor_to_array

from examples.probability.model import Config, State, Trial, classify
from examples.probability.vectors import Diagnostics, Kinematics, Layers, VectorRenderer, contact_components


def object_xml(kind: str) -> ET.Element:
    root = ET.Element("mujoco", model=f"probability_{kind}")
    ET.SubElement(root, "compiler", angle="radian", inertiafromgeom="false")
    body = ET.SubElement(ET.SubElement(root, "worldbody"), "body", name="object")
    ET.SubElement(body, "freejoint")
    if kind == "die":
        # Uniform 80 mm cube, 80 g. Explicit inertia excludes the decorative pips.
        ET.SubElement(body, "inertial", pos="0 0 0", mass="0.08", diaginertia="0.0000853333333 " * 3)
        ET.SubElement(body, "geom", type="box", size="0.04 0.04 0.04", rgba="0.93 0.91 0.83 1")
        layouts = (
            [(0, 0)], [(-1, -1), (1, 1)], [(-1, -1), (0, 0), (1, 1)],
            [(-1, -1), (-1, 1), (1, -1), (1, 1)],
            [(-1, -1), (-1, 1), (0, 0), (1, -1), (1, 1)],
            [(-1, -1), (-1, 0), (-1, 1), (1, -1), (1, 0), (1, 1)],
        )
        normals = [(0, 0, 1), (1, 0, 0), (0, 1, 0), (0, -1, 0), (-1, 0, 0), (0, 0, -1)]
        for normal, layout in zip(normals, layouts):
            axis = int(np.argmax(np.abs(normal)))
            tangent_axes = [i for i in range(3) if i != axis]
            for u, v in layout:
                pos = np.array(normal, dtype=float) * 0.036
                pos[tangent_axes] = np.array([u, v]) * 0.019
                ET.SubElement(body, "geom", type="sphere", size="0.007", pos=" ".join(map(str, pos)),
                              rgba="0.03 0.06 0.09 1", mass="0", contype="0", conaffinity="0")
    else:
        # Uniform cylinder: radius 45 mm, thickness 8 mm, mass 20 g.
        ET.SubElement(body, "inertial", pos="0 0 0", mass="0.02",
                      diaginertia="0.0000102316667 0.0000102316667 0.00002025")
        ET.SubElement(body, "geom", type="cylinder", size="0.045 0.004", rgba="0.73 0.49 0.16 1")
        for side, color in ((1, "0.95 0.73 0.26 1"), (-1, "0.16 0.62 0.75 1")):
            ET.SubElement(body, "geom", type="cylinder", size="0.043 0.0001", pos=f"0 0 {side * 0.0041}",
                          rgba=color, mass="0", contype="0", conaffinity="0")
        # H on +Z, T on -Z. Bars are visual only, so the mass and contacts remain symmetric.
        for x, y, sx, sy, side in ((-.014, 0, .003, .022, 1), (.014, 0, .003, .022, 1),
                                    (0, 0, .014, .003, 1), (0, .019, .022, .003, -1),
                                    (0, -.002, .003, .021, -1)):
            ET.SubElement(body, "geom", type="box", size=f"{sx} {sy} 0.0001",
                          pos=f"{x} {y} {side * 0.0043}", rgba="0.03 0.06 0.09 1",
                          mass="0", contype="0", conaffinity="0")
    return root


class World:
    def __init__(self, config: Config, count: int, render: bool = False):
        self.config = config
        self.count = count
        self.render = render
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=config.dt, substeps=2, gravity=(0, 0, -9.81)),
            rigid_options=gs.options.RigidOptions(
                enable_torsional_friction=True, enable_rolling_friction=True,
            ),
            show_viewer=False,
            vis_options=gs.options.VisOptions(ambient_light=(0.5, 0.5, 0.5), rendered_envs_idx=[0]),
        )
        tile = np.empty((64, 64, 3), dtype=np.uint8)
        tile[:] = [33, 49, 61]
        tile[:32, :32] = tile[32:, 32:] = [38, 56, 68]
        self.floor = self.scene.add_entity(
            morph=gs.morphs.Plane(tile_size=(0.2, 0.2)),
            material=gs.materials.Rigid(friction=config.friction, friction_rolling=0.0001, friction_torsional=0.005),
            surface=gs.surfaces.Default(diffuse_texture=gs.textures.ImageTexture(image_array=tile)),
        )
        self.body = self.scene.add_entity(
            morph=gs.morphs.MJCF(file=object_xml(config.kind)),
            material=gs.materials.Rigid(friction=config.friction, friction_rolling=0.0001,
                                       friction_torsional=0.005),
        )
        self.camera = None
        if render:
            self.camera = self.scene.add_camera(res=(960, 540), pos=(0.9, -1.25, 1.05),
                                                lookat=(0, 0, 0.22), fov=38, GUI=False, debug=True)
        self.scene.build(n_envs=count)
        self.local_base_index = self.body.base_link_idx - self.body.link_start
        self.vector_renderer = VectorRenderer(self.scene, self.body.geoms[0]) if render else None
        self.still = np.zeros(count, dtype=int)
        self.ticks = 0
        self.elapsed = 0.0
        self.trail = []

    def reset(self, state: State):
        self.scene.reset()
        self.body.set_pos(state.position)
        self.body.set_quat(state.quaternion)
        self.body.set_dofs_velocity(state.velocity)
        self.still[:] = 0
        self.ticks = 0
        self.elapsed = 0.0
        self.initial_kinematics = self.kinematics()
        self.trail = [self.initial_kinematics.center.copy()]

    def kinematics(self) -> Kinematics:
        center = tensor_to_array(self.body.get_links_pos(
            [self.local_base_index], ref=gs.link_ref_frame.link_COM))[0, 0].copy()
        linear = tensor_to_array(self.body.get_links_vel(
            [self.local_base_index], ref=gs.link_ref_frame.link_COM))[0, 0].copy()
        return Kinematics(center, tensor_to_array(self.body.get_quat())[0].copy(), linear,
                          tensor_to_array(self.body.get_ang())[0].copy())

    def diagnostics(self) -> Diagnostics:
        current = self.kinematics()
        floor_origin = tensor_to_array(self.floor.get_pos())[0].copy()
        floor_rotation = Rotation.from_quat(tensor_to_array(self.floor.get_quat())[0], scalar_first=True)
        floor_normal = floor_rotation.apply(self.floor.morph.normal)
        floor_linear = tensor_to_array(self.floor.get_vel())[0]
        floor_angular = tensor_to_array(self.floor.get_ang())[0]
        floor_friction = self.floor.geoms[0].get_friction().item()
        entity_friction = self.body.geoms[0].get_friction().item()
        # Genesis mixes a rigid contact's sliding coefficients by max(a, b, 0.01).
        friction = max(floor_friction, entity_friction, 0.01)
        contacts = []
        if self.ticks:
            raw = self.body.get_contacts(with_entity=self.floor)
            valid = tensor_to_array(raw["valid_mask"])[0]
            positions = tensor_to_array(raw["position"])[0][valid]
            normals = tensor_to_array(raw["normal"])[0][valid]
            geom_a = tensor_to_array(raw["geom_a"])[0][valid]
            entity_is_a = (geom_a >= self.body.geom_start) & (geom_a < self.body.geom_end)
            force_a = tensor_to_array(raw["force_a"])[0][valid]
            force_b = tensor_to_array(raw["force_b"])[0][valid]
            forces = np.where(entity_is_a[:, None], force_a, force_b)
            normals = np.where(entity_is_a[:, None], normals, -normals)
            for position, normal, force in zip(positions, normals, forces):
                floor_velocity = floor_linear + np.cross(floor_angular, position - floor_origin)
                contacts.append(contact_components(position, normal, force, current.center,
                                                   current.linear, current.angular, floor_velocity))
        initial = self.initial_kinematics
        return Diagnostics(current.center, current.quaternion, current.linear, current.angular,
                           initial.center, initial.quaternion, initial.linear, initial.angular,
                           floor_origin, floor_normal, floor_friction, entity_friction, friction,
                           contacts, self.elapsed, np.array(self.trail))

    def state(self) -> State:
        return State(tensor_to_array(self.body.get_pos()).copy(), tensor_to_array(self.body.get_quat()).copy(),
                     tensor_to_array(self.body.get_dofs_velocity()).copy())

    def step(self) -> tuple[np.ndarray, np.ndarray]:
        self.scene.step(update_visualizer=False)
        self.ticks += 1
        self.elapsed = self.ticks * self.config.dt
        velocity = tensor_to_array(self.body.get_dofs_velocity())
        pose = tensor_to_array(self.body.get_pos())
        if self.render:
            # Both authored entities place the link origin at their COM; keep every physical path sample.
            self.trail.append(pose[0].copy())
        # Require contact-height proximity too: an instant of slow motion near an apex is not rest.
        quiet = (np.linalg.norm(velocity[:, :3], axis=1) < 0.015)
        quiet &= np.linalg.norm(velocity[:, 3:], axis=1) < 0.15
        quiet &= pose[:, 2] < 0.065
        self.still = np.where(quiet, self.still + 1, 0)
        at_rest = self.still * self.config.dt >= 0.25
        faces = classify(tensor_to_array(self.body.get_quat()), self.config.kind)
        return at_rest, faces

    def image(self, diagnostics: Diagnostics | None = None, layers: Layers | None = None) -> np.ndarray:
        if diagnostics is not None:
            self.vector_renderer.draw(diagnostics, Layers() if layers is None else layers)
        rgb, _, _, _ = self.camera.render()
        # Batched rendering may retain the one rendered environment dimension.
        return rgb[0] if rgb.ndim == 4 else rgb

    def run(self, initial: State, seeds: list[int], progress: Callable[[float], bool] | None = None,
            horizon: float | None = None) -> list[Trial]:
        rollout = self.run_steps(initial, seeds, horizon)
        step = 0
        while True:
            try:
                elapsed = next(rollout)
            except StopIteration as finished:
                return finished.value
            if step % 20 == 0 and progress is not None and not progress(elapsed):
                rollout.close()
                return []
            step += 1

    def run_steps(self, initial: State, seeds: list[int],
                  horizon: float | None = None) -> Generator[float, None, list[Trial]]:
        """Yield between physics steps so the worker can keep drawing the visible toss."""
        horizon = self.config.horizon if horizon is None else horizon
        self.reset(initial)
        outcomes = np.full(self.count, -1, dtype=int)
        ended = np.zeros(self.count, dtype=bool)
        times = np.full(self.count, horizon)
        final = State(initial.position.copy(), initial.quaternion.copy(), initial.velocity.copy())
        for _ in range(round(horizon / self.config.dt)):
            resting, faces = self.step()
            newly_ended = resting & ~ended
            if newly_ended.any():
                current = self.state()
                outcomes[newly_ended] = faces[newly_ended]
                times[newly_ended] = self.elapsed
                final.position[newly_ended] = current.position[newly_ended]
                final.quaternion[newly_ended] = current.quaternion[newly_ended]
                final.velocity[newly_ended] = current.velocity[newly_ended]
                ended |= newly_ended
            if ended.all():
                break
            yield self.elapsed
        current = self.state()
        final.position[~ended] = current.position[~ended]
        final.quaternion[~ended] = current.quaternion[~ended]
        final.velocity[~ended] = current.velocity[~ended]
        return [Trial(seed, int(outcomes[i]), float(times[i]), initial.position[i].tolist(),
                      initial.quaternion[i].tolist(), initial.velocity[i].tolist(), final.position[i].tolist(),
                      final.quaternion[i].tolist(), final.velocity[i].tolist()) for i, seed in enumerate(seeds)]

    def close(self):
        self.scene.destroy()
