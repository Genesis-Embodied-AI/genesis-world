"""Visualize the raycaster's ``ray_alignment`` (base / yaw / world).

Three carrier boxes hang side by side over a box obstacle each. A dense ray
grid (0.2 m resolution) hangs from each carrier. The carriers move in two
phases so each pair of alignment modes is compared where it differs most:

Phase 1 - pitch swing (about x, amplitude 60 deg, 1.0 Hz)
    ``base`` tilts its grid with the body, ``yaw``/``world`` stay level.
    Highlights the base vs yaw/world difference.

Phase 2 - yaw rotation (about z, the yaw axis)
    ``base`` and ``yaw`` rotate their grid footprint with the body, ``world``
    keeps the grid fixed in the world frame. Highlights the yaw vs world
    difference.

The rays point straight down (0, 0, -1), so in phase 2 the ray grid's
rectangular footprint on the ground rotates for ``base``/``yaw`` while it
stays put for ``world``.

Each raycaster draws its rays and hit points (``draw_debug``).

Run:  python examples/sensors/raycaster_alignment.py
Add ``--headless`` to run without a window, and ``--seconds N`` to auto-stop.
"""

import argparse
import math
import os
import time

import genesis as gs
from genesis.utils.geom import euler_to_quat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true", help="Run without the viewer")
    parser.add_argument("--seconds", type=float, default=0.0, help="Auto-stop after N seconds (0 = run until closed)")
    parser.add_argument("--phase1", type=float, default=8.0, help="Pitch-swing phase length (s)")
    parser.add_argument("--phase2", type=float, default=8.0, help="Yaw-rotation phase length (s)")
    parser.add_argument("--swing", type=float, default=60.0, help="Pitch swing amplitude (deg)")
    parser.add_argument("--yaw_rate", type=float, default=90.0, help="Yaw rotation rate (deg/s)")
    args = parser.parse_args()

    gs.init(logging_level="warning")
    scene = gs.Scene(
        vis_options=gs.options.VisOptions(show_world_frame=True),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(-2.0, -7.0, 6.0),
            camera_lookat=(0.0, 0.0, 1.0),
        ),
        profiling_options=gs.options.ProfilingOptions(show_FPS=False),
        show_viewer=not args.headless,
    )

    # Ground plane (link 0); one box obstacle (1x1x1, top at z=1.0) per column.
    scene.add_entity(gs.morphs.Plane())

    columns = ("base", "yaw", "world")
    xs = {"base": -3.0, "yaw": 0.0, "world": 3.0}
    obstacles = {
        label: scene.add_entity(gs.morphs.Box(size=(1.0, 1.0, 1.0), pos=(xs[label], 0.0, 0.5), fixed=True))
        for label in columns
    }
    carriers = {
        label: scene.add_entity(gs.morphs.Box(size=(0.2, 0.2, 0.2), pos=(xs[label], 0.0, 3.0), fixed=True))
        for label in columns
    }

    rcs = {}
    for label in columns:
        rcs[label] = scene.add_sensor(
            gs.sensors.Raycaster(
                pattern=gs.sensors.raycaster.GridPattern(resolution=0.2, size=(2.0, 2.0), direction=(0.0, 0.0, -1.0)),
                entity_idx=carriers[label].idx,
                max_range=5.0,
                ray_alignment=label,
                return_points=True,
                return_world_frame=True,
                draw_debug=True,
                debug_ray_start_color=(0.0, 0.0, 0.0, 0.0),
                debug_ray_hit_color=(1.0, 0.0, 0.0, 1.0),
                pos_offset=(0.0, 0.0, -0.5),
            )
        )

    scene.build(n_envs=1)
    for _ in range(3):
        scene.step()

    print("\nRaycaster ray_alignment demo (two-phase motion)")
    print("-" * 64)
    print(f"Phase 1 ({args.phase1}s): pitch swing, amplitude {args.swing:.0f} deg, 1.0 Hz")
    print("  -> compares base (tilts) vs yaw/world (level)")
    print(f"Phase 2 ({args.phase2}s): yaw rotation, {args.yaw_rate:.0f} deg/s")
    print("  -> compares yaw (grid follows) vs world (grid fixed)")

    print("\nClose the viewer window (or Ctrl+C) to exit.")
    try:
        start = time.time()
        t = 0.0
        while True:
            if t < args.phase1:
                # Phase 1: pitch swing about x at 0.5 Hz.
                pitch = math.radians(args.swing) * math.sin(2.0 * math.pi * 1.0 * t)
                quat = euler_to_quat((pitch, 0.0, 0.0))
            else:
                # Phase 2: continuous yaw rotation about z.
                yaw = math.radians(args.yaw_rate) * (t - args.phase1)
                quat = euler_to_quat((0.0, 0.0, yaw))
            for carrier in carriers.values():
                carrier.set_quat(quat, relative=False)
            scene.step()
            t += scene.dt
            if args.seconds and time.time() - start > args.seconds:
                break
            if "PYTEST_VERSION" in os.environ:
                break
    except KeyboardInterrupt:
        gs.logger.info("Simulation interrupted, exiting.")
    finally:
        gs.logger.info("Simulation finished.")


if __name__ == "__main__":
    main()
