"""
Interactive SurfaceDistanceProbe demo with Shadow Hand and keyboard teleop.

Surface distance probes on the hand measure distance to a rubber duck (mesh) and a box.
Use keyboard controls to move the hand via IK; the hand tracks target positions
for the wrist and fingertips.
"""

import argparse
import os

import numpy as np

import genesis as gs
import genesis.utils.geom as gu
from genesis.vis.keybindings import Key, KeyAction, Keybind

# Teleop
KEY_DPOS = 0.015
FORCE_SCALE = 10.0

# Surface distance probe
MAX_RANGE = 0.5

# Objects
DUCK_POS = (-0.2, 0.4, 0.6)
DUCK_QUAT = gu.euler_to_quat((90.0, 0.0, -90.0))
BOX_POS = (-0.3, 0.2, 0.4)
BOX_QUAT = (1, 0, 0, 0)


def main():
    parser = argparse.ArgumentParser(description="Interactive SurfaceDistanceProbe with Shadow Hand")
    parser.add_argument("-v", "--vis", action="store_true", help="Show visualization GUI")
    parser.add_argument("-g", "--gpu", action="store_true", help="Run on GPU instead of CPU")
    parser.add_argument("-t", "--seconds", type=float, default=3.0, help="Seconds to simulate (headless mode)")
    args = parser.parse_args()

    gs.init(
        backend=gs.gpu if args.gpu else gs.cpu,
        precision="32",
        logging_level="info",
    )

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            gravity=(0.0, 0.0, 0.0),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(-1.2, 0.6, 1.0),
            camera_lookat=(0.0, 0.0, 0.5),
        ),
        show_viewer=args.vis,
    )

    hand_pos_init = np.array([0.0, 0.0, 0.35], dtype=np.float32)
    hand_quat_init = gu.euler_to_quat((0.0, 0.0, -90.0))
    robot = scene.add_entity(
        morph=gs.morphs.URDF(
            file="urdf/shadow_hand/shadow_hand.urdf",
            pos=hand_pos_init,
            quat=hand_quat_init,
        ),
    )
    duck = scene.add_entity(
        material=gs.materials.Rigid(),
        morph=gs.morphs.Mesh(
            file="meshes/duck.obj",
            scale=0.06,
            pos=DUCK_POS,
            quat=DUCK_QUAT,
        ),
        surface=gs.surfaces.Default(
            color=(0.95, 0.75, 0.2, 1.0),
        ),
    )
    box = scene.add_entity(
        material=gs.materials.Rigid(),
        morph=gs.morphs.Box(
            size=(0.06, 0.06, 0.08),
            pos=BOX_POS,
        ),
        surface=gs.surfaces.Default(
            color=(0.3, 0.7, 0.4, 1.0),
        ),
    )

    sensors = []
    for link_name in (
        "palm",
        "index_finger_distal",
        "middle_finger_distal",
        "ring_finger_distal",
        "little_finger_distal",
        "thumb_distal",
    ):
        sensor = scene.add_sensor(
            gs.sensors.SurfaceDistanceProbe(
                entity_idx=robot.idx,
                link_idx_local=robot.get_link(link_name).idx_local,
                probe_local_pos=((0.0, 0.0, 0.0),),
                probe_radius=MAX_RANGE,
                track_link_idx=(duck.base_link_idx, box.base_link_idx),
                draw_debug=args.vis,
            )
        )
        sensors.append(sensor)

    scene.build()

    hand_pos = hand_pos_init.copy()
    if args.vis:
        for obj in (duck, box):
            obj.set_dofs_kv((0.8, 0.8, 0.8, 0.02, 0.02, 0.02))
            obj.control_dofs_position(0.0)
        robot.set_dofs_kp(FORCE_SCALE / KEY_DPOS)
        robot.set_dofs_kv(0.1 * FORCE_SCALE / KEY_DPOS)
        robot.control_dofs_position(robot.get_dofs_position())

    # Register keybindings
    is_running = True
    if args.vis:

        def stop():
            nonlocal is_running
            is_running = False

        def translate(index: int, is_negative: bool):
            hand_pos[index] += (-1 if is_negative else 1) * KEY_DPOS

        def reset_pose():
            robot.set_pos(hand_pos_init)
            duck.set_pos(DUCK_POS)
            duck.set_quat(DUCK_QUAT)
            box.set_pos(BOX_POS)
            box.set_quat(BOX_QUAT)

        scene.viewer.register_keybinds(
            Keybind("move_forward", Key.UP, KeyAction.HOLD, callback=translate, args=(0, False)),
            Keybind("move_back", Key.DOWN, KeyAction.HOLD, callback=translate, args=(0, True)),
            Keybind("move_right", Key.RIGHT, KeyAction.HOLD, callback=translate, args=(1, True)),
            Keybind("move_left", Key.LEFT, KeyAction.HOLD, callback=translate, args=(1, False)),
            Keybind("move_down", Key.J, KeyAction.HOLD, callback=translate, args=(2, True)),
            Keybind("move_up", Key.K, KeyAction.HOLD, callback=translate, args=(2, False)),
            Keybind("reset", Key.BACKSLASH, KeyAction.RELEASE, callback=reset_pose),
            Keybind("quit", Key.ESCAPE, KeyAction.RELEASE, callback=stop),
        )

    print("\n=== SurfaceDistanceProbe with Shadow Hand ===")
    print("Surface distance probes on hand palm and fingertips, tracking duck and box links")
    if args.vis:
        print("Keyboard: [Up/Down/Left/Right] move hand XY, [n/m] up/down, [\\] reset, [ESC] quit")
    else:
        print(f"Running headless for {args.seconds}s ...")
    print()

    # Simulation loop
    try:
        while is_running:
            if args.vis:
                robot.control_dofs_position(hand_pos, dofs_idx_local=slice(0, 3))
            else:
                distances = []
                for sensor in sensors:
                    distances.append(sensor.read())
                print(f"Surface distances: {distances}")

            scene.step()

            if "PYTEST_VERSION" in os.environ:
                break
            if not args.vis and scene.get_time() > args.seconds:
                break
    except KeyboardInterrupt:
        gs.logger.info("Simulation interrupted.")
    finally:
        gs.logger.info("Simulation finished.")


if __name__ == "__main__":
    main()
