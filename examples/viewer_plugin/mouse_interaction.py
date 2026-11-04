import argparse
import math
import os

import genesis as gs
from genesis.vis.keybindings import Key, KeyAction, Keybind


def main():
    parser = argparse.ArgumentParser(description="Mouse interaction viewer plugin example.")
    parser.add_argument("--use-force", action="store_true", help="Apply spring forces instead of setting position")
    parser.add_argument(
        "--use-visual-geom",
        action="store_true",
        help="Grab entities by their visual mesh instead of their collision one",
    )
    parser.add_argument("-b", "--num-envs", type=int, default=1, help="Number of parallel environments")
    args = parser.parse_args()

    gs.init(backend=gs.cpu)

    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3.5, 0.0, 2.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=40,
        ),
        show_viewer=True,
    )

    scene.add_entity(
        gs.morphs.Plane(),
    )

    # Only entities opting into visual raycasting can be grabbed with --use-visual-geom.
    raycastable = gs.materials.Rigid(
        use_visual_raycasting=True,
    )
    vis_mode = "visual" if args.use_visual_geom else "collision"

    scene.add_entity(
        morph=gs.morphs.Sphere(
            pos=(-0.3, -0.3, 0),
            radius=0.1,
        ),
        material=raycastable,
        vis_mode=vis_mode,
    )
    scene.add_entity(
        morph=gs.morphs.Mesh(
            file="meshes/duck/duck.obj",
            pos=(0.0, 0.0, 0.5),
            scale=0.001,
        ),
        material=raycastable,
        vis_mode=vis_mode,
    )
    for i in range(6):
        angle = i * (2 * math.pi / 6)
        radius = 0.5 + i * 0.1
        scene.add_entity(
            morph=gs.morphs.Box(
                pos=(radius * math.cos(angle), radius * math.sin(angle), 0.1 + i * 0.1),
                size=(0.2, 0.2, 0.2),
            ),
            material=raycastable,
            vis_mode=vis_mode,
        )

    scene.viewer.add_plugin(
        gs.vis.viewer_plugins.MouseInteractionPlugin(
            use_force=args.use_force,
            color=(0.1, 0.6, 0.8, 0.6),
            use_visual_geom=args.use_visual_geom,
        )
    )

    scene.build(n_envs=args.num_envs)

    is_running = True

    def stop():
        nonlocal is_running
        is_running = False

    scene.viewer.register_keybinds(
        Keybind("quit", Key.ESCAPE, KeyAction.RELEASE, callback=stop),
    )

    try:
        while is_running:
            scene.step()

            if "PYTEST_VERSION" in os.environ:
                break
    except KeyboardInterrupt:
        gs.logger.info("Simulation interrupted, exiting.")
    finally:
        gs.logger.info("Simulation finished.")


if __name__ == "__main__":
    main()
