import argparse

import numpy as np

import genesis as gs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", help="Show visualization GUI")
    args = parser.parse_args()

    gs.init(backend=gs.cpu, precision="32", logging_level="info")
    np.set_printoptions(precision=7, suppress=True)

    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.0, -2, 1.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=40,
        ),
        show_viewer=args.vis,
    )

    robot = scene.add_entity(
        morph=gs.morphs.Mesh(
            file="meshes/duck.obj",
            scale=0.06,
            pos=(3.5, -1.5, 0.7),
        ),
        material=gs.materials.Kinematic(),
    )

    target_entity = scene.add_entity(
        gs.morphs.Mesh(
            file="meshes/axis.obj",
            scale=0.15,
            collision=False,
        ),
        material=gs.materials.Kinematic(),
        surface=gs.surfaces.Default(
            color=(1, 0.5, 0.5, 1),
        ),
    )

    scene.build()

    target_quat = np.array([0, 1, 0, 0])  # pointing downwards
    center = np.array([0.4, -0.2, 0.25])
    r = 0.1
    ee_link = robot.links[0]

    for i in range(0, 2000):
        target_pos = center + np.array([np.cos(i / 360 * np.pi), np.sin(i / 360 * np.pi), 0]) * r

        target_entity.set_qpos(np.concatenate([target_pos, target_quat]))
        q = robot.inverse_kinematics(
            link=ee_link,
            pos=target_pos,
            quat=target_quat,
        )

        # Note that this IK example is only for visualizing the solved q, so here we do not call scene.step(), but only update the state and the visualizer
        # In actual control applications, you should instead use robot.control_dofs_position() and scene.step()
        robot.set_qpos(q)
        scene.visualizer.update(force=True)


if __name__ == "__main__":
    main()
