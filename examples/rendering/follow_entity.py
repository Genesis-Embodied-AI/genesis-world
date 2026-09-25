import argparse

import genesis as gs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", help="Show visualization GUI")
    parser.add_argument("--fix", action="store_true", help="Keep the camera orientation fixed while following")
    args = parser.parse_args()

    gs.init(backend=gs.cpu)

    scene = gs.Scene(
        vis_options=gs.options.VisOptions(
            rendered_envs_idx=(0,),
        ),
        profiling_options=gs.options.ProfilingOptions(
            show_FPS=False,
        ),
        show_viewer=False,
    )
    scene.add_entity(morph=gs.morphs.Plane())
    cube = scene.add_entity(
        gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(0.0, -0.9, 1.0),
            euler=(15.0, 30.0, 60.0),
        )
    )

    cam = scene.add_camera(
        res=(640, 480),
        pos=(2.0, 0.0, 1.5),
        lookat=(0, 0, 0.7),
        fov=40,
        GUI=args.vis,
    )
    cam.follow_entity(cube, fix_orientation=args.fix)

    scene.build()

    cube.set_dofs_velocity([0.0, 5.0, 0.0, 0.0, 0.0, 1.0])
    for _ in range(100):
        scene.step()
        cam.render()


if __name__ == "__main__":
    main()
