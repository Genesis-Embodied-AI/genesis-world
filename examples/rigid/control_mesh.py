import argparse

import genesis as gs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", help="Show visualization GUI")
    args = parser.parse_args()

    gs.init(backend=gs.cpu)

    viewer_options = gs.options.ViewerOptions(
        camera_pos=(0, -3.5, 2.5),
        camera_lookat=(0.0, 0.0, 0.5),
        camera_fov=40,
    )

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
            gravity=(0, 0, 0),
        ),
        vis_options=gs.options.VisOptions(
            show_link_frame=True,
        ),
        viewer_options=viewer_options,
        show_viewer=args.vis,
    )

    duck = scene.add_entity(
        morph=gs.morphs.Mesh(
            file="meshes/duck.obj",
            scale=0.1,
            pos=(0, 0, 0.0),
        ),
    )
    scene.build()

    dofs_idx = duck.base_joint.dofs_idx
    duck.set_dofs_kp((0.3,) * 6, dofs_idx)
    duck.set_dofs_kv((1.0,) * 6, dofs_idx)

    pos = duck.get_dofs_position()
    pos[-1] = 1.0  # rotate around intrinsic z axis
    duck.control_dofs_position(pos, dofs_idx)

    for i in range(1000):
        scene.step()


if __name__ == "__main__":
    main()
