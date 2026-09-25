import argparse
import genesis as gs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", help="Show visualization GUI")
    args = parser.parse_args()

    gs.init(backend=gs.cpu)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0, -3.5, 2.5),
            camera_lookat=(0.0, 0.0, 1.0),
            camera_fov=40,
        ),
        show_viewer=args.vis,
        show_FPS=False,
    )

    plane = scene.add_entity(
        gs.morphs.Plane(),
    )
    cube = scene.add_entity(
        gs.morphs.Box(
            pos=(0, 0, 1.0),
            size=(0.2, 0.2, 0.2),
        ),
    )
    scene.build(n_envs=1)

    link_idx = [1]
    rotation_direction = 1
    for i in range(1000):
        cube_pos = scene.sim.rigid_solver.get_links_pos(link_idx)
        cube_pos[:, :, 2] -= 1
        scene.sim.rigid_solver.apply_links_external_wrench(
            force=-100 * cube_pos, torque=[0, 0, rotation_direction * 5], links_idx=link_idx
        )

        scene.step()

        if (i + 50) % 100 == 0:
            rotation_direction *= -1


if __name__ == "__main__":
    main()
