import argparse
import os

import numpy as np

import genesis as gs
from genesis.utils.geom import euler_to_quat
from genesis.vis.keybindings import Key, KeyAction, Keybind

# Position and angle increments for keyboard teleop control
KEY_DPOS = 0.1
KEY_DANGLE = 0.1

# Number of obstacles to create in a ring around the robot
NUM_CYLINDERS = 8
NUM_BOXES = 6
CYLINDER_RING_RADIUS = 3.0
BOX_RING_RADIUS = 5.0


def main():
    parser = argparse.ArgumentParser(description="Genesis LiDAR/Depth Camera Visualization with Keyboard Teleop")
    parser.add_argument("-b", "--num-envs", type=int, default=0, help="Number of environments to replicate")
    parser.add_argument("-g", "--gpu", action="store_true", help="Run on GPU instead of CPU")
    parser.add_argument("--use-box", action="store_true", help="Use Box as robot instead of Go2")
    parser.add_argument(
        "--pattern", type=str, default="spherical", choices=("spherical", "depth", "grid"), help="Sensor pattern type"
    )
    args = parser.parse_args()

    gs.init(backend=gs.gpu if args.gpu else gs.cpu, precision="32", logging_level="info")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            gravity=(0.0, 0.0, -1.0),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(-6.0, 0.0, 4.0),
            camera_lookat=(0.0, 0.0, 0.5),
        ),
        profiling_options=gs.options.ProfilingOptions(
            show_FPS=False,
        ),
        show_viewer=True,
    )

    scene.add_entity(gs.morphs.Plane())

    # create ring of obstacles to visualize raycaster sensor hits
    for i in range(NUM_CYLINDERS):
        angle = 2 * np.pi * i / NUM_CYLINDERS
        x = CYLINDER_RING_RADIUS * np.cos(angle)
        y = CYLINDER_RING_RADIUS * np.sin(angle)
        scene.add_entity(
            gs.morphs.Cylinder(
                height=1.5,
                radius=0.3,
                pos=(x, y, 0.75),
                fixed=True,
            )
        )

    for i in range(NUM_BOXES):
        angle = 2 * np.pi * i / NUM_BOXES + np.pi / 6
        x = BOX_RING_RADIUS * np.cos(angle)
        y = BOX_RING_RADIUS * np.sin(angle)
        scene.add_entity(
            gs.morphs.Box(
                size=(0.5, 0.5, 2.0 * (i + 1) / NUM_BOXES),
                pos=(x, y, 1.0),
                fixed=False,
            )
        )

    entity_kwargs = dict(
        pos=(0.0, 0.0, 0.35),
        quat=(1.0, 0.0, 0.0, 0.0),
        fixed=True,
    )

    if args.use_box:
        robot = scene.add_entity(
            gs.morphs.Box(
                size=(0.1, 0.1, 0.1),
                **entity_kwargs,
            )
        )
        pos_offset = (0.0, 0.0, 0.2)
    else:
        robot = scene.add_entity(
            gs.morphs.URDF(
                file="urdf/go2/urdf/go2.urdf",
                **entity_kwargs,
            )
        )
        pos_offset = (0.3, 0.0, 0.1)

    sensor_kwargs = dict(
        entity_idx=robot.idx,
        pos_offset=pos_offset,
        euler_offset=(0.0, 0.0, 0.0),
        return_world_frame=True,
        draw_debug=True,
    )

    if args.pattern == "depth":
        sensor = scene.add_sensor(gs.sensors.DepthCamera(pattern=gs.sensors.DepthCameraPattern(), **sensor_kwargs))
        scene.add_recorder(
            data_func=(lambda: sensor.read_image()[0]) if args.num_envs > 0 else sensor.read_image,
            rec_options=gs.recorders.MPLImagePlot(),
        )
    else:
        if args.pattern == "grid":
            pattern_cfg = gs.sensors.GridPattern()
        else:
            if args.pattern != "spherical":
                gs.logger.warning(f"Unrecognized raycaster pattern: {args.pattern}. Using 'spherical' instead.")
            pattern_cfg = gs.sensors.SphericalPattern()

        sensor = scene.add_sensor(gs.sensors.Lidar(pattern=pattern_cfg, **sensor_kwargs))

    scene.build(n_envs=args.num_envs)

    # Initialize pose state
    init_pos = np.array([0.0, 0.0, 0.35], dtype=np.float32)
    init_euler = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    target_pos = init_pos.copy()
    target_euler = init_euler.copy()

    def apply_pose_to_all_envs(pos_np: np.ndarray, quat_np: np.ndarray):
        if args.num_envs > 0:
            pos_np = np.expand_dims(pos_np, axis=0).repeat(args.num_envs, axis=0)
            quat_np = np.expand_dims(quat_np, axis=0).repeat(args.num_envs, axis=0)
        robot.set_pos(pos_np)
        robot.set_quat(quat_np)

    # Define control callbacks
    def reset_pose():
        target_pos[:] = init_pos
        target_euler[:] = init_euler

    def translate(index: int, is_negative: bool):
        target_pos[index] += (-1 if is_negative else 1) * KEY_DPOS

    def rotate(index: int, is_negative: bool):
        target_euler[index] += (-1 if is_negative else 1) * KEY_DANGLE

    # Register keybindings
    scene.viewer.register_keybinds(
        Keybind("move_forward", Key.UP, KeyAction.HOLD, callback=translate, args=(0, False)),
        Keybind("move_backward", Key.DOWN, KeyAction.HOLD, callback=translate, args=(0, True)),
        Keybind("move_right", Key.RIGHT, KeyAction.HOLD, callback=translate, args=(1, True)),
        Keybind("move_left", Key.LEFT, KeyAction.HOLD, callback=translate, args=(1, False)),
        Keybind("move_down", Key.J, KeyAction.HOLD, callback=translate, args=(2, True)),
        Keybind("move_up", Key.K, KeyAction.HOLD, callback=translate, args=(2, False)),
        Keybind("roll_ccw", Key.N, KeyAction.HOLD, callback=rotate, args=(0, False)),
        Keybind("roll_cw", Key.M, KeyAction.HOLD, callback=rotate, args=(0, True)),
        Keybind("pitch_up", Key.COMMA, KeyAction.HOLD, callback=rotate, args=(1, False)),
        Keybind("pitch_down", Key.PERIOD, KeyAction.HOLD, callback=rotate, args=(1, True)),
        Keybind("yaw_ccw", Key.O, KeyAction.HOLD, callback=rotate, args=(2, False)),
        Keybind("yaw_cw", Key.P, KeyAction.HOLD, callback=rotate, args=(2, True)),
        Keybind("reset", Key.BACKSLASH, KeyAction.RELEASE, callback=reset_pose),
        overwrite=True,
    )

    print("Keyboard Controls:")
    print("[Up/Down/Left/Right]: Move XY")
    print("[j/k]: Down/Up")
    print("[n/m]: Roll CCW/CW")
    print("[,/.]: Pitch Up/Down")
    print("[o/p]: Yaw CCW/CW")
    print("[\\]: Reset")

    apply_pose_to_all_envs(target_pos, euler_to_quat(target_euler))

    try:
        while True:
            apply_pose_to_all_envs(target_pos, euler_to_quat(target_euler))
            scene.step()

            if "PYTEST_VERSION" in os.environ:
                break
    except KeyboardInterrupt:
        gs.logger.info("Simulation interrupted, exiting.")
    finally:
        gs.logger.info("Simulation finished.")


if __name__ == "__main__":
    main()
