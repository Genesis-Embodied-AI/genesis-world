import argparse
import os

import genesis as gs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--object",
        type=str,
        default="cylinder",
        choices=("sphere", "cylinder", "duck"),
        help="Shape stacked into the tower",
    )
    parser.add_argument("-v", "--vis", action="store_true", help="Show visualization GUI")
    args = parser.parse_args()
    object_type = args.object
    horizon = 20 if "PYTEST_VERSION" in os.environ else 10000

    gs.init(backend=gs.cpu, precision="32", performance_mode=True)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.0015,
        ),
        rigid_options=gs.options.RigidOptions(
            max_collision_pairs=400,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(20, -20, 20),
            camera_lookat=(0.0, 0.0, 5.0),
        ),
        show_viewer=args.vis,
    )

    scene.add_entity(gs.morphs.Plane())

    # create pyramid of boxes
    box_width, box_length, box_height = 0.25, 2.0, 0.1
    num_stacks = 50
    for i in range(num_stacks):
        if i % 2 == 0:  # horizontal stack
            box_size = (box_width, box_length, box_height)
            box_pos_0 = (-0.4 * box_length, 0, i * (box_height - 1e-3) + 0.5 * box_height)
            box_pos_1 = (+0.4 * box_length, 0, i * (box_height - 1e-3) + 0.5 * box_height)
        else:  # vertical stack
            box_size = (box_length, box_width, box_height)
            box_pos_0 = (0, -0.4 * box_length, i * (box_height - 1e-3) + 0.5 * box_height)
            box_pos_1 = (0, +0.4 * box_length, i * (box_height - 1e-3) + 0.5 * box_height)
        for box_pos in (box_pos_0, box_pos_1):
            scene.add_entity(
                gs.morphs.Box(
                    size=box_size,
                    pos=box_pos,
                ),
            )

    # Drop a huge mesh
    if object_type == "duck":
        duck_scale = 0.8
        scene.add_entity(
            morph=gs.morphs.Mesh(
                file="meshes/duck.obj",
                scale=duck_scale,
                pos=(0, -0.1, num_stacks * box_height + 10 * duck_scale),
            ),
        )
    elif object_type == "sphere":
        sphere_radius = 2.0
        scene.add_entity(
            morph=gs.morphs.Sphere(
                radius=sphere_radius,
                pos=(0.0, 0.0, num_stacks * box_height + 5 * sphere_radius),
            ),
        )
    elif object_type == "cylinder":
        cylinder_radius, cylinder_height = 2.0, 1.0
        scene.add_entity(
            morph=gs.morphs.Cylinder(
                radius=cylinder_radius,
                height=cylinder_height,
                pos=(0.0, 0.0, num_stacks * box_height + 5 * cylinder_height),
            ),
        )

    scene.build()
    for i in range(horizon):
        scene.step()


if __name__ == "__main__":
    main()
