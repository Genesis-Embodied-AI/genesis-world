"""Closed-loop linkages built from MJCF equality constraints.

A four-bar linkage cannot be expressed as a tree of joints, so the loop is closed by an equality constraint.
Both spellings MuJoCo offers are available here: `connect` pins two bodies at a point, `weld` also locks their
relative orientation.
"""

import argparse

import genesis as gs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", help="Show visualization GUI")
    parser.add_argument("-s", "--steps", type=int, default=1000, help="Number of simulation steps")
    parser.add_argument(
        "--constraint",
        type=str,
        default="weld",
        choices=("connect", "weld"),
        help="Equality constraint closing the loop",
    )
    args = parser.parse_args()

    gs.init(backend=gs.cpu)

    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(10, 0, 10),
            camera_lookat=(0.0, 0.0, 3),
            camera_fov=60,
        ),
        show_viewer=args.vis,
    )
    linkage = scene.add_entity(
        gs.morphs.MJCF(
            file="xml/four_bar_linkage.xml" if args.constraint == "connect" else "xml/four_bar_linkage_weld.xml",
        ),
    )

    scene.build()

    # Start away from the rest pose so the loop visibly swings instead of sitting in equilibrium.
    qpos = linkage.get_qpos()
    qpos[:3] = 0.2
    linkage.set_qpos(qpos)

    for _ in range(args.steps):
        scene.step()


if __name__ == "__main__":
    main()
