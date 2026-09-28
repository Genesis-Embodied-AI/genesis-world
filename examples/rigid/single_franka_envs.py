import argparse

import genesis as gs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", help="Show visualization GUI")
    parser.add_argument("-g", "--gpu", action="store_true", help="Run on GPU instead of CPU")
    parser.add_argument("--sep", action="store_true", help="Simulate each environment as a separate island")
    parser.add_argument("-r", "--record", action="store_true", help="Record the scene to video")
    parser.add_argument("-b", "--num-envs", type=int, default=1, help="Number of parallel environments")
    parser.add_argument("-s", "--steps", type=int, default=100, help="Number of simulation steps")
    args = parser.parse_args()

    gs.init(backend=gs.gpu if args.gpu else gs.cpu)

    scene = gs.Scene(
        vis_options=gs.options.VisOptions(
            show_world_frame=True,
            show_link_frame=True,
            plane_reflection=True,
            split_envs=args.sep,
            rendered_envs_idx=list(range(args.num_envs)),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3.5, 0.0, 2.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=40,
        ),
        show_viewer=args.vis,
    )

    plane = scene.add_entity(
        gs.morphs.Plane(),
    )
    franka = scene.add_entity(
        gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
        ),
        visualize_contact=True,
    )

    cam_0 = scene.add_camera(
        res=(1280, 960),
        pos=(3.5, 0.0, 2.5),
        lookat=(0, 0, 0.5),
        fov=30,
        GUI=args.vis,
    )
    scene.build(n_envs=args.num_envs, env_spacing=(0.5, 0.5))

    if args.record:
        cam_0.start_recording(save_to_filename="out/single_franka_envs.mp4")

    for i in range(args.steps):
        scene.step()

        color, depth, seg, normal = cam_0.render(
            rgb=True, depth=True, segmentation=True, colorize_seg=True, normal=True
        )
        print(f"Step {i}:", args.num_envs, color.shape, depth.shape, seg.shape, normal.shape)

    if args.record:
        cam_0.stop_recording()


if __name__ == "__main__":
    main()
