"""
Interactive TemperatureGrid sensor visualization with keyboard teleop.

A platform has a temperature grid sensor on its top surface. Move a "hot" pusher
and drop objects onto the platform; the grid shows temperature (blue=cool, red=hot)
from contact-based blending of each body's base_temperature and conductivity.
"""

import argparse
import os

import numpy as np

import genesis as gs
from genesis.utils.misc import tensor_to_array
from genesis.vis.keybindings import Key, KeyAction, Keybind

# Teleop
KEY_DPOS = 0.08
KEY_DPOS_Z = 0.01
FORCE_SCALE = 100.0
PUSHER_SIZE = 0.1

# Temperature grid
GRID_SIZE = (10, 10, 1)

# Objects
SANDBOX_SIZE = 1.5
WALL_THICKNESS = 0.08
WALL_HEIGHT = 0.3
PLATFORM_HEIGHT = 0.1
OBJ_Z = PLATFORM_HEIGHT * 1.4
OBJ_SIZE = 0.1


def main():
    parser = argparse.ArgumentParser(description="Interactive TemperatureGrid sensor visualization")
    parser.add_argument("-v", "--vis", action="store_true", help="Show visualization GUI")
    parser.add_argument("-g", "--gpu", action="store_true", help="Run on GPU instead of CPU")
    parser.add_argument("-t", "--seconds", type=float, default=3.0, help="Seconds to simulate (headless mode)")
    parser.add_argument("--simulate-all-links", action="store_true", help="Simulate all link temperatures")
    args = parser.parse_args()

    gs.init(
        backend=gs.gpu if args.gpu else gs.cpu,
        precision="32",
        logging_level="info",
    )

    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(-SANDBOX_SIZE * 2, 0.0, PLATFORM_HEIGHT + 2.5),
            camera_lookat=(0.0, 0.0, PLATFORM_HEIGHT),
        ),
        profiling_options=gs.options.ProfilingOptions(
            show_FPS=False,
        ),
        show_viewer=args.vis,
    )

    scene.add_entity(gs.morphs.Plane())

    # Sandbox: four fixed walls
    for i in range(4):
        scene.add_entity(
            gs.morphs.Box(
                size=(SANDBOX_SIZE + 2 * WALL_THICKNESS, WALL_THICKNESS, WALL_HEIGHT),
                pos=(
                    *((1 - 2 * (i % 2)) * (SANDBOX_SIZE / 2 + WALL_THICKNESS / 2), 0.0)[:: (2 * (i >= 2) - 1)],
                    WALL_HEIGHT / 2,
                ),
                euler=(0, 0, 90 * (i >= 2)),
                fixed=True,
            ),
            surface=gs.surfaces.Default(
                color=(0.5, 0.45, 0.4, 1.0),
            ),
        )

    platform = scene.add_entity(
        gs.morphs.Box(
            size=(SANDBOX_SIZE, SANDBOX_SIZE, PLATFORM_HEIGHT),
            pos=(0.0, 0.0, PLATFORM_HEIGHT / 2),
            fixed=True,
            visualization=False,  # sensor debug_draw will be shown
        ),
        material=gs.materials.Rigid(
            friction=0.01,
        ),
    )

    pusher_pos_init = np.array([0.0, 0.0, PLATFORM_HEIGHT + PUSHER_SIZE / 2], dtype=np.float32)
    pusher = scene.add_entity(
        gs.morphs.Cylinder(
            radius=PUSHER_SIZE,
            height=PUSHER_SIZE,
            pos=pusher_pos_init,
        ),
        surface=gs.surfaces.Default(
            color=(1.0, 0.2, 0.2, 1.0),
        ),
        material=gs.materials.Rigid(
            friction=0.01,
        ),
    )

    rect = scene.add_entity(
        gs.morphs.Box(
            size=(OBJ_SIZE, OBJ_SIZE * 2, OBJ_SIZE),
            pos=(SANDBOX_SIZE / 4, 0, OBJ_Z),
        ),
    )
    cylinder = scene.add_entity(
        gs.morphs.Cylinder(
            radius=OBJ_SIZE / 2,
            height=OBJ_SIZE * 1.2,
            pos=(0, SANDBOX_SIZE / 4, OBJ_Z),
        ),
    )
    sphere = scene.add_entity(
        gs.morphs.Sphere(
            radius=OBJ_SIZE / 2,
            pos=(-SANDBOX_SIZE / 4, -SANDBOX_SIZE / 4, OBJ_Z),
        ),
    )
    objects = [rect, cylinder, sphere]

    # Build properties_dict: one TemperatureProperties per link (required for contact blending).
    # Platform is room temp; pusher is hot; objects are warm; ground plane is cool.
    properties_dict = {
        -1: gs.sensors.TemperatureProperties(
            base_temperature=-40.0,
            conductivity=200.0,
            density=2000.0,
            specific_heat=1.0,
            emissivity=0.85,
        ),
        platform.base_link_idx: gs.sensors.TemperatureProperties(
            base_temperature=22.0,
            conductivity=100.0,
            density=1000.0,
            specific_heat=0.2,
            emissivity=0.4,
        ),
        pusher.base_link_idx: gs.sensors.TemperatureProperties(
            base_temperature=200.0,
            conductivity=1000.0,
            density=2000.0,
            specific_heat=1.0,
            emissivity=0.8,
        ),
    }

    temperature_sensor = scene.add_sensor(
        gs.sensors.TemperatureGrid(
            entity_idx=platform.idx,
            link_idx_local=0,
            grid_size=GRID_SIZE,
            properties_dict=properties_dict,
            draw_debug=args.vis,
            debug_temperature_range=(0.0, 80.0),
            simulate_all_link_temperatures=args.simulate_all_links,
            ambient_temperature=22.0,
            convection_coefficient=0.0,
        )
    )

    scene.build()

    if args.vis:
        pusher.set_dofs_kp(FORCE_SCALE / KEY_DPOS, dofs_idx_local=slice(0, 3))
        pusher.set_dofs_kv(0.1 * FORCE_SCALE / KEY_DPOS, dofs_idx_local=slice(0, 3))
        pusher.control_dofs_position(pusher.get_dofs_position())

    # Register keybindings
    is_running = True
    if args.vis:
        target_pos = pusher_pos_init.copy()
        obj_idx = 0

        def stop():
            nonlocal is_running
            is_running = False

        def reset_pose():
            target_pos[:] = pusher_pos_init
            pusher.set_dofs_position(pusher_pos_init, dofs_idx_local=slice(0, 3))

        def translate(index: int, is_negative: bool):
            target_pos[index] += (-1 if is_negative else 1) * (KEY_DPOS if index < 2 else KEY_DPOS_Z)

        def drop_object():
            nonlocal obj_idx
            drop_pos = pusher.get_pos()
            drop_pos[2] += 0.2 * (obj_idx + 1)
            objects[obj_idx].set_pos(drop_pos)
            objects[obj_idx].set_quat(np.array([1, 0, 0, 0], dtype=np.float32))
            obj_idx = (obj_idx + 1) % len(objects)

        scene.viewer.register_keybinds(
            Keybind("move_forward", Key.UP, KeyAction.HOLD, callback=translate, args=(0, False)),
            Keybind("move_backward", Key.DOWN, KeyAction.HOLD, callback=translate, args=(0, True)),
            Keybind("move_right", Key.RIGHT, KeyAction.HOLD, callback=translate, args=(1, True)),
            Keybind("move_left", Key.LEFT, KeyAction.HOLD, callback=translate, args=(1, False)),
            Keybind("move_down", Key.J, KeyAction.HOLD, callback=translate, args=(2, True)),
            Keybind("move_up", Key.K, KeyAction.HOLD, callback=translate, args=(2, False)),
            Keybind("drop_object", Key.SPACE, KeyAction.PRESS, callback=drop_object),
            Keybind("reset", Key.BACKSLASH, KeyAction.RELEASE, callback=reset_pose),
            Keybind("quit", Key.ESCAPE, KeyAction.RELEASE, callback=stop),
        )

    print("\n=== Interactive TemperatureGrid ===")
    print(f"Platform {SANDBOX_SIZE}m x {SANDBOX_SIZE}m with grid {GRID_SIZE}")
    if args.vis:
        print()
        print("Keyboard Controls:")
        print("  [Up/Down/Left/Right]  Move pusher (hot) in XY")
        print("  [j / k]     Lower / raise pusher")
        print("  [SPACE]     Drop an object at pusher location")
        print("  [\\]         Reset pusher position")
    else:
        print(f"Running headless for {args.seconds}s ...")
    print()

    # Simulation loop
    try:
        while is_running:
            if args.vis:
                pusher.control_dofs_position(target_pos, dofs_idx_local=slice(0, 3))

            scene.step()

            if args.vis:
                cur_pos = tensor_to_array(pusher.get_pos())
                target_pos[:] = np.clip(target_pos - cur_pos, -KEY_DPOS, KEY_DPOS) + cur_pos

            data = temperature_sensor.read()
            t_min, t_max = float(data.min()), float(data.max())
            print(f"time={scene.get_time():.2f}s: Temperature range [{t_min:.1f}, {t_max:.1f}] degC")
            if args.simulate_all_links:
                print(f"Link temperatures: {temperature_sensor.link_temperatures}")

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
