import numpy as np
import genesis as gs
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pile-type",
        type=str,
        default="static",
        choices=("static", "falling"),
        help="Whether the pile starts settled or falls into place",
    )
    parser.add_argument(
        "--num-cubes",
        type=int,
        default=5,
        choices=(5, 6, 7, 8, 9, 10),
        help="Number of cubes along the base of the pyramid",
    )
    parser.add_argument("-g", "--gpu", action="store_true", help="Run on GPU instead of CPU")
    parser.add_argument("-s", "--steps", type=int, default=150, help="Number of simulation steps")
    parser.add_argument("-v", "--vis", action="store_true", help="Show visualization GUI")
    args = parser.parse_args()

    gs.init(backend=gs.gpu if args.gpu else gs.cpu, precision="32")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0, -5.5, 2.5),
            camera_lookat=(0, 0.0, 1.5),
        ),
        show_viewer=args.vis,
    )

    plane = scene.add_entity(gs.morphs.Plane())

    # create pyramid of boxes
    box_size = 0.25
    box_spacing = (1.0 - 1e-3 + 0.1 * (args.pile_type == "falling")) * box_size
    box_pos_offset = (-0.5, 1, 0.0) + 0.5 * np.array([box_size, box_size, box_size])
    boxes = {}
    for i in range(args.num_cubes):
        for j in range(args.num_cubes - i):
            box = scene.add_entity(
                gs.morphs.Box(
                    size=[box_size, box_size, box_size],
                    pos=box_pos_offset + box_spacing * np.array([i + 0.5 * j, 0, j]),
                ),
            )

    scene.build()

    for i in range(args.steps):
        scene.step()


if __name__ == "__main__":
    main()
