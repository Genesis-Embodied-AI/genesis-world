"""
Interactive demo of tactile sensors on a fixed taxel pad (box or dome) with controllable objects.
Sensor types: ContactDepthProbe, ContactProbe, ElastomerTaxel, KinematicTaxel, ProximityTaxel.

Note that the sensor readings here have not been calibrated to any units, and is purely for visualization purposes.
"""

import argparse
import os
import tempfile
from typing import TYPE_CHECKING

import numpy as np
import torch
import trimesh

import genesis as gs
import genesis.utils.geom as gu
from genesis.recorders.plotters import IS_MATPLOTLIB_AVAILABLE
from genesis.utils.misc import tensor_to_array
from genesis.vis.keybindings import Key, KeyAction, Keybind

if TYPE_CHECKING:
    from genesis.engine.entities.rigid_entity import RigidEntity
    from genesis.engine.sensors.base_sensor import Sensor

KEY_DPOS = 0.001
FORCE_SCALE = 100.0
ROT_FORCE_SCALE = 100.0

GRID_SIZE = 20  # 20x20 taxels for square
PROBE_RADIUS = 0.004
OBJECT_INITIAL_CLEARANCE = 0.01
OBJECT_MAX_PENETRATION = 1e-3

SENSOR_OBJ_SIZE = 0.15
SENSOR_OBJ_Z = 0.05
OBJECT_SIZE = 0.08

OBJ_PER_ENV_LABELS = ("torus", "sphere", "duck", "dragon")


def _add_tactile_sensor(
    scene: gs.Scene,
    entity: "RigidEntity",
    link_idx_local: int,
    sensor_type: str,
    probe_local_pos: np.ndarray,
    probe_normal: tuple[float, float, float] | np.ndarray,
    track_link_idx: tuple[int, ...],
    contact_depth_query: str | None,
    noise: bool,
) -> "Sensor":
    common = dict(
        entity_idx=entity.idx,
        link_idx_local=link_idx_local,
        draw_debug=True,
        probe_radius=PROBE_RADIUS,
        contact_depth_query=contact_depth_query,
    )
    if noise:
        common.update(
            hysteresis_strength=0.5,
            hysteresis_tau=0.1,  # seconds
            probe_radius_noise=0.001,  # meters
            probe_gain=1.5,
        )
    if sensor_type == "elastomer":
        return scene.add_sensor(
            gs.sensors.ElastomerTaxel(
                probe_local_pos=probe_local_pos,
                probe_local_normal=probe_normal,
                track_link_idx=track_link_idx,
                n_sample_points=1000,
                lambda_d=5000.0,
                lambda_s=4000.0,
                dilate_scale=0.1,
                shear_scale=1.0,
                normal_exponent=1.5,
                compressibility=0.8,
                debug_point_cloud_radius=0.001,
                **common,
            )
        )

    grid_local_pos = probe_local_pos  # (ny, nx, 3) for the plane grid; flattened below for the non-grid sensors
    is_grid = probe_local_pos.ndim == 3
    probe_local_pos = probe_local_pos.reshape(-1, 3)
    # Spatial crosstalk needs a regular grid layout, so enable it under --noise only for the grid-capable taxel
    # sensors (and only the plane grid, not the dome). The 3x3 kernel sums to 1, so it conserves total force.
    grid_crosstalk_kwargs = (
        dict(
            probe_local_pos=grid_local_pos,
            crosstalk_kernel=[[0.03, 0.07, 0.03], [0.07, 0.60, 0.07], [0.03, 0.07, 0.03]],
        )
        if noise and is_grid
        else dict(probe_local_pos=probe_local_pos)
    )
    if sensor_type == "depth":
        return scene.add_sensor(
            gs.sensors.ContactDepthProbe(
                probe_local_pos=probe_local_pos,
                **common,
            )
        )
    if sensor_type == "contact":
        return scene.add_sensor(
            # Schmitt-trigger thresholds (contact depth in meters): a taxel latches on above contact_threshold and
            # only releases once the depth drops back below the lower release_threshold.
            gs.sensors.ContactProbe(
                probe_local_pos=probe_local_pos,
                contact_threshold=0.004,
                release_threshold=0.002,
                **common,
            )
        )
    if sensor_type == "kinematic":
        return scene.add_sensor(
            gs.sensors.KinematicTaxel(
                normal_stiffness=500.0,
                normal_damping=1.0,
                shear_scalar=4.0,
                twist_scalar=1.0,
                normal_exponent=1.5,
                **grid_crosstalk_kwargs,
                **common,
            )
        )

    common["probe_radius"] = PROBE_RADIUS * 5
    if sensor_type == "proximity":
        return scene.add_sensor(
            gs.sensors.ProximityTaxel(
                track_link_idx=track_link_idx,
                n_sample_points=4000,
                stiffness=40.0,
                shear_coupling=10.0,
                probe_local_normal=probe_normal,
                debug_point_cloud_radius=0.0005,
                debug_probe_sphere_opacity=0.0,
                debug_probe_color=(0.2, 0.6, 1.0),
                debug_contact_color=(1.0, 0.2, 0.2),
                **grid_crosstalk_kwargs,
                **common,
            )
        )
    raise ValueError(sensor_type)


def _force_torque_field(sensor: "Sensor", n_envs: int) -> tuple[np.ndarray, np.ndarray]:
    reading = sensor.read()
    force = tensor_to_array(reading.force).reshape(n_envs, -1, 3)
    torque = tensor_to_array(reading.torque).reshape(n_envs, -1, 3)
    return force, torque


def _plot_tactile_sensor(
    scene: gs.Scene,
    sensor_type: str,
    sensor: "Sensor",
    n_envs: int = 1,
    plot_normal: tuple[float, float, float] = (0.0, 0.0, -1.0),
) -> None:
    """Set up a single live plot window: one vector-field subplot per environment for the per-taxel sensors, or one
    line plot with a line per environment for the scalar (depth / contact-count) sensors."""
    if not IS_MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available; skipping plot setup.")
        return

    env_titles = OBJ_PER_ENV_LABELS[:n_envs]

    # read_field returns (n_envs, N, 3) for the force/displacement field. The two taxel sensors that also estimate a
    # torque return a (force, torque) pair, which overlays a curved twist arrow (twist_scale_factor set).
    vector_field_setup = {
        "elastomer": (
            "ElastomerTaxel marker displacements",
            1.0,
            0.01,
            None,
            1.0,
            lambda: tensor_to_array(sensor.read()).reshape(n_envs, -1, 3),
        ),
        "kinematic": (
            "KinematicTaxel force + twist",
            0.01,
            1.0,
            0.002,
            2.0,
            lambda: _force_torque_field(sensor, n_envs),
        ),
        "proximity": (
            "ProximityTaxel force + twist",
            0.5,
            2.0,
            0.001,
            1.0,
            lambda: _force_torque_field(sensor, n_envs),
        ),
    }
    if sensor_type in vector_field_setup:
        title, scale_factor, max_magnitude, twist_scale_factor, twist_max_magnitude, read_field = vector_field_setup[
            sensor_type
        ]
        scene.add_recorder(
            read_field,
            gs.recorders.MPLVectorFieldPlot(
                title=title,
                positions=sensor.probe_local_pos.reshape(-1, 3),
                normal=plot_normal,
                scale_factor=scale_factor,
                max_magnitude=max_magnitude,
                subplot_titles=env_titles,
                twist_scale_factor=twist_scale_factor,
                twist_max_magnitude=twist_max_magnitude,
            ),
        )
        return

    # Scalar sensors: one line per env in a single plot. data_func returns one value per env.
    title, y_label, reduce_fn = {
        "depth": ("ContactDepthProbe max depth", "depth", lambda r: float(r.max())),
        "contact": ("ContactProbe taxels in contact", "# taxels", lambda r: float(r.sum())),
    }[sensor_type]
    scene.add_recorder(
        lambda: tuple(reduce_fn(sensor.read()[i]) for i in range(n_envs)),
        gs.recorders.MPLLinePlot(
            labels=env_titles,
            x_label="step",
            y_label=y_label,
            history_length=200,
            title=title,
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive tactile sandbox with selectable sensor type")
    parser.add_argument("-v", "--vis", action="store_true", help="Show visualization GUI")
    parser.add_argument("-g", "--gpu", action="store_true", help="Run on GPU instead of CPU")
    parser.add_argument(
        "--set-pos", action="store_true", help="Set object position directly instead of using control force"
    )
    parser.add_argument("-t", "--seconds", type=float, default=3.0, help="Seconds to simulate (headless mode)")
    parser.add_argument("--dome", action="store_true", help="Change the sensor object to a dome instead of a box")
    parser.add_argument(
        "--sensor",
        choices=("elastomer", "depth", "contact", "kinematic", "proximity"),
        default="elastomer",
        help="Tactile sensor implementation",
    )
    parser.add_argument(
        "--contact-depth-query",
        choices=("sdf", "raycast"),
        default=None,
        help="Contact-depth backend, falling back to the sensor's own default when unset",
    )
    parser.add_argument(
        "--noise",
        action="store_true",
        help="Enable sensor imperfections (viscoelastic hysteresis, probe_radius_noise, probe_gain, and spatial "
        "crosstalk on grid taxel sensors)",
    )
    args = parser.parse_args()

    gs.init(
        backend=gs.gpu if args.gpu else gs.cpu,
        precision="32",
        logging_level="info",
    )

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            substeps=4,
            gravity=(0.0, 0.0, 0.0),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.5, -0.2, 0.5),
            camera_lookat=(0.0, 0.0, SENSOR_OBJ_Z),
        ),
        profiling_options=gs.options.ProfilingOptions(
            show_FPS=True,
        ),
        show_viewer=args.vis,
    )

    scene.add_entity(gs.morphs.Plane())

    if args.dome:
        sensor_morph = gs.morphs.Sphere(
            radius=SENSOR_OBJ_SIZE / 2,
            pos=(0.0, 0.0, -SENSOR_OBJ_SIZE / 2 + SENSOR_OBJ_Z),
            fixed=True,
        )
    else:
        sensor_morph = gs.morphs.Box(
            size=(SENSOR_OBJ_SIZE, SENSOR_OBJ_SIZE, SENSOR_OBJ_Z),
            pos=(0.0, 0.0, SENSOR_OBJ_Z / 2),
            fixed=True,
        )
    sensor_obj = scene.add_entity(
        morph=sensor_morph,
        surface=gs.surfaces.Default(
            color=(0.8, 0.8, 0.8, 1.0),
        ),
        material=gs.materials.Rigid(
            friction=0.6,
        ),
    )
    probe_normal_axis = (0.0, 0.0, 1.0)
    if args.dome:
        sphere_radius = SENSOR_OBJ_SIZE / 2
        probe_local_pos, probe_normal = gu.generate_ring_points_on_sphere(
            radius=sphere_radius,
            cap_axis=probe_normal_axis,
            n_rings=GRID_SIZE,
            arc_spacing=2.0 * PROBE_RADIUS,
            return_normals=True,
        )
    else:
        probe_normal = probe_normal_axis
        probe_z = SENSOR_OBJ_Z / 2
        probe_local_pos = gu.generate_grid_points_on_plane(
            lo=(-SENSOR_OBJ_SIZE / 2, -SENSOR_OBJ_SIZE / 2, probe_z),
            hi=(SENSOR_OBJ_SIZE / 2, SENSOR_OBJ_SIZE / 2, probe_z),
            normal=probe_normal_axis,
            nx=GRID_SIZE,
            ny=GRID_SIZE,
        )

    torus_path = os.path.join(tempfile.gettempdir(), "tactile_sandbox_torus.obj")
    if not os.path.exists(torus_path):
        trimesh.creation.torus(major_radius=1.0, minor_radius=0.5).export(torus_path)

    obj = scene.add_entity(
        morph=[
            gs.morphs.Mesh(
                file=torus_path,
                scale=OBJECT_SIZE / 2,
                convexify=False,
            ),
            gs.morphs.Sphere(
                radius=OBJECT_SIZE / 2,
            ),
            gs.morphs.Mesh(
                file="meshes/duck.obj",
                euler=(90.0, 0.0, 0.0),
                scale=0.03,
            ),
            gs.morphs.Mesh(
                file="meshes/dragon/dragon.obj",
                euler=(90.0, 0.0, 90.0),
                scale=0.001,
            ),
        ],
        surface=gs.surfaces.Default(
            color=(1.0, 1.0, 1.0, 1.0),
        ),
        material=gs.materials.Rigid(
            friction=0.5,
        ),
    )

    sensor = _add_tactile_sensor(
        scene,
        sensor_obj,
        0,
        args.sensor,
        probe_local_pos,
        probe_normal,
        track_link_idx=(obj.base_link_idx,),
        contact_depth_query=args.contact_depth_query,
        noise=args.noise,
    )
    if args.vis and "PYTEST_VERSION" not in os.environ:
        _plot_tactile_sensor(scene, args.sensor, sensor, n_envs=4, plot_normal=probe_normal_axis)
    scene.build(n_envs=4, env_spacing=(SENSOR_OBJ_SIZE * 1.2, SENSOR_OBJ_SIZE * 1.2))

    obj_init_pos = tensor_to_array(obj.get_pos())
    obj_init_quat = tensor_to_array(obj.get_quat())
    obj_aabb = tensor_to_array(obj.get_vAABB())
    obj_init_pos[..., 2] += SENSOR_OBJ_Z + OBJECT_INITIAL_CLEARANCE - obj_aabb[..., 0, 2]
    obj.set_pos(obj_init_pos)

    obj_target_pos = obj_init_pos.copy()
    obj_target_euler = gu.quat_to_xyz(obj_init_quat, rpy=True, degrees=True)
    obj_target_quat = obj_init_quat.copy()
    obj_contact_pos = obj_init_pos.copy()
    obj_contact_pos[..., 2] -= OBJECT_INITIAL_CLEARANCE + OBJECT_MAX_PENETRATION

    is_running = True

    if args.vis:
        obj.set_dofs_kp(FORCE_SCALE / KEY_DPOS, dofs_idx_local=slice(0, 3))
        obj.set_dofs_kp(ROT_FORCE_SCALE / KEY_DPOS, dofs_idx_local=slice(3, 6))
        obj.set_dofs_kv(0.1 * FORCE_SCALE / KEY_DPOS, dofs_idx_local=slice(0, 6))

        def stop():
            nonlocal is_running
            is_running = False

        def reset_pose():
            nonlocal obj_target_pos, obj_target_euler, obj_target_quat
            obj_target_pos = obj_init_pos.copy()
            obj_target_euler = gu.quat_to_xyz(obj_init_quat, rpy=True, degrees=True)
            obj_target_quat = obj_init_quat.copy()
            obj.set_pos(obj_init_pos)
            obj.set_quat(obj_init_quat)

        def translate(index: int, is_negative: bool):
            nonlocal obj_target_pos
            delta = (-1 if is_negative else 1) * KEY_DPOS
            obj_target_pos[..., index] += delta

        def rotate(axis_idx: int, is_negative: bool):
            nonlocal obj_target_euler, obj_target_quat
            delta = -5.0 if is_negative else 5.0
            obj_target_euler[..., axis_idx] += delta
            obj_target_quat = gu.euler_to_quat(obj_target_euler)

        scene.viewer.register_keybinds(
            Keybind("move_forward", Key.UP, KeyAction.HOLD, callback=translate, args=(0, True)),
            Keybind("move_backward", Key.DOWN, KeyAction.HOLD, callback=translate, args=(0, False)),
            Keybind("move_right", Key.RIGHT, KeyAction.HOLD, callback=translate, args=(1, False)),
            Keybind("move_left", Key.LEFT, KeyAction.HOLD, callback=translate, args=(1, True)),
            Keybind("move_down", Key.J, KeyAction.HOLD, callback=translate, args=(2, True)),
            Keybind("move_up", Key.K, KeyAction.HOLD, callback=translate, args=(2, False)),
            Keybind("rotate_cw", Key.M, KeyAction.HOLD, callback=rotate, args=(2, True)),
            Keybind("rotate_ccw", Key.N, KeyAction.HOLD, callback=rotate, args=(2, False)),
            Keybind("rotate_roll_left", Key.COMMA, KeyAction.HOLD, callback=rotate, args=(0, True)),
            Keybind("rotate_roll_right", Key.PERIOD, KeyAction.HOLD, callback=rotate, args=(0, False)),
            Keybind("reset", Key.BACKSLASH, KeyAction.RELEASE, callback=reset_pose),
            Keybind("quit", Key.ESCAPE, KeyAction.RELEASE, callback=stop),
        )

    print("\n=== Tactile Sensor Sandbox ===")
    n_taxels = probe_local_pos.reshape(-1, 3).shape[0]
    layout = f"dome ({GRID_SIZE} latitude rings)" if args.dome else f"plane grid {probe_local_pos.shape[:-1]}"
    print(f"sensor={args.sensor}; taxels={n_taxels}; {layout}; noise={'on' if args.noise else 'off'}")
    if args.vis and IS_MATPLOTLIB_AVAILABLE:
        print("Matplotlib live plot enabled when supported.")
    if args.vis:
        print()
        print("Keyboard Controls:")
        print("  [UP/DOWN/LEFT/RIGHT] Move selected object in XY")
        print("  [j / k]              Lower / raise selected object")
        print("  [n / m]              Rotate object around Z axis")
        print("  [SPACE]              Select next object")
        print("  [\\]                  Reset objects")
    else:
        obj.set_pos(obj_contact_pos)
        print(f"Running headless for {args.seconds}s ...")
    print()

    try:
        while is_running:
            # Every environment of this scene runs the same schedule, so one of them stands for the clock printed
            # and compared below.
            t = float(scene.get_time(envs_idx=0))
            if args.vis:
                if args.set_pos:
                    obj.set_pos(obj_target_pos)
                    obj.set_quat(obj_target_quat)
                else:
                    obj.control_dofs_position(gu.quat_to_xyz(obj_target_quat), dofs_idx_local=slice(3, 6))
                    obj.control_dofs_position(obj_target_pos, dofs_idx_local=slice(0, 3))
            data = sensor.read()
            if args.sensor == "elastomer":
                magnitude = torch.linalg.norm(data, dim=-1)
                if magnitude.max() > gs.EPS:
                    print(f"t={t:.2f}s  max|displacement|={magnitude.max():.5f}")
            elif args.sensor == "depth":
                max_depth = data.max()
                if max_depth > gs.EPS:
                    print(f"t={t:.2f}s  max depth={max_depth:.4f}")
            elif args.sensor == "contact":
                n_contact = int(data.sum())
                if n_contact > 0:
                    print(f"t={t:.2f}s  taxels in contact={n_contact}")
            elif args.sensor in ("kinematic", "proximity"):
                # |twist| is the plotted torque about the view normal (the twist_scalar spin term); print to calibrate.
                force_mag = torch.linalg.norm(data.force, dim=-1).max()
                twist_mag = (data.torque @ data.torque.new_tensor(probe_normal_axis)).abs().max()
                if force_mag > gs.EPS:
                    print(f"t={t:.2f}s  max|F|={force_mag:.4f}  max|twist|={twist_mag:.5f}")

            scene.step()

            if "PYTEST_VERSION" in os.environ:
                break
            if not args.vis and t >= args.seconds:
                break
    except KeyboardInterrupt:
        gs.logger.info("Simulation interrupted.")
    finally:
        gs.logger.info("Simulation finished.")


if __name__ == "__main__":
    main()
