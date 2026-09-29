"""Visualize the raycaster's ``exclude_link_idx`` on a Unitree Go2.

Two identical hovering Go2 robots sit side by side. Each has a downward grid
raycaster on its base, so the two cases never overlap in the viewer:

- left  ``no_exclude``: rays hit the robot's own legs (short distances, hit
  points on the body), so the scan reads the robot itself instead of the
  terrain.
- right ``exclude_go2``: ``exclude_link_idx`` removes that robot's every link
  from the cast, so its rays pass through the body and read the ground height
  below it, the behaviour a terrain height scan needs.

Each raycaster draws its rays and hit points (``draw_debug``).

Run:  python examples/sensors/raycaster_exclude_go2.py
Add ``--headless`` to run without a window, and ``--seconds N`` to auto-stop.
"""

import argparse
import os
import time

import numpy as np

import genesis as gs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true", help="Run without the viewer")
    parser.add_argument("--seconds", type=float, default=0.0, help="Auto-stop after N seconds (0 = run until closed)")
    args = parser.parse_args()

    gs.init(logging_level="warning")
    scene = gs.Scene(
        vis_options=gs.options.VisOptions(show_world_frame=True),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.0, -7.0, 5.0),
            camera_lookat=(0.0, 0.0, 1.0),
        ),
        profiling_options=gs.options.ProfilingOptions(show_FPS=False),
        show_viewer=not args.headless,
    )

    scene.add_entity(gs.morphs.Plane())

    # Two identical hovering Go2 robots, one per raycaster.
    def make_go2(x):
        return scene.add_entity(
            gs.morphs.URDF(
                file="urdf/go2/urdf/go2.urdf",
                pos=(x, 0.0, 0.8),
                fixed=True,
            ),
        )

    go2_no = make_go2(-2.0)
    go2_ex = make_go2(2.0)
    go2_ex_link_idx = [link.idx for link in go2_ex.links]

    def grid_raycaster(entity, exclude):
        return scene.add_sensor(
            gs.sensors.Raycaster(
                pattern=gs.sensors.raycaster.GridPattern(resolution=0.1, size=(1.2, 0.6), direction=(0.0, 0.0, -1.0)),
                entity_idx=entity.idx,
                max_range=3.0,
                ray_alignment="world",
                exclude_link_idx=exclude,
                return_points=True,
                return_world_frame=True,
                draw_debug=True,
                debug_ray_start_color=(0.0, 0.0, 0.0, 0.0),
                debug_ray_hit_color=(1.0, 0.0, 0.0, 1.0),
            )
        )

    rcs = {
        "no_exclude": grid_raycaster(go2_no, ()),
        "exclude_go2": grid_raycaster(go2_ex, go2_ex_link_idx),
    }

    scene.build(n_envs=1)
    for _ in range(3):
        scene.step()

    print("\nRaycaster exclude_link_idx demo (Go2)")
    print("-" * 64)
    print(f"{'sensor':14s} {'x':6s} {'min_dist':10s} {'mean_dist':10s} {'hits_on_robot'}")
    for name, (x, rc) in (("no_exclude", (-2.0, rcs["no_exclude"])), ("exclude_go2", (2.0, rcs["exclude_go2"]))):
        data = rc.read()
        dist = data.distances.cpu().numpy().reshape(-1)
        pts = data.points.cpu().numpy().reshape(-1, 3)
        # A hit on the robot sits well above the ground (z >> 0); ground hits
        # cluster near z=0.
        robot_hits = int((pts[:, 2] > 0.2).sum())
        hit = dist[dist < 2.9]
        print(f"{name:14s} {x:6.1f} {hit.min():10.3f} {hit.mean():10.3f} {robot_hits:14d}")

    print("\nViewer legend (color-coded hit points):")
    print("  left  no_exclude : rays hit the robot's own legs (red on the body)")
    print("  right exclude    : rays pass through the robot and read the ground")

    print("\nClose the viewer window (or Ctrl+C) to exit.")
    try:
        start = time.time()
        while True:
            scene.step()
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
