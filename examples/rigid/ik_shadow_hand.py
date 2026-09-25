import argparse

import numpy as np

import genesis as gs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", help="Show visualization GUI")
    args = parser.parse_args()

    gs.init(backend=gs.cpu, precision="32", logging_level="info")

    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(2.5, 0.0, 1.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=40,
        ),
        show_viewer=args.vis,
    )

    target_1 = scene.add_entity(
        gs.morphs.Mesh(
            file="meshes/axis.obj",
            scale=0.05,
        ),
        material=gs.materials.Kinematic(),
        surface=gs.surfaces.Default(
            color=(1, 0.5, 0.5, 1),
        ),
    )

    target_2 = scene.add_entity(
        gs.morphs.Mesh(
            file="meshes/axis.obj",
            scale=0.05,
        ),
        material=gs.materials.Kinematic(),
        surface=gs.surfaces.Default(
            color=(0.5, 1.0, 0.5, 1),
        ),
    )
    target_3 = scene.add_entity(
        gs.morphs.Mesh(
            file="meshes/axis.obj",
            scale=0.05,
        ),
        material=gs.materials.Kinematic(),
        surface=gs.surfaces.Default(
            color=(0.5, 0.5, 1.0, 1),
        ),
    )
    robot = scene.add_entity(
        morph=gs.morphs.URDF(
            scale=1.0,
            file="urdf/shadow_hand/shadow_hand.urdf",
        ),
        material=gs.materials.Kinematic(),
        surface=gs.surfaces.Reflective(
            color=(0.4, 0.4, 0.4),
        ),
    )

    scene.build()
    scene.reset()

    target_quat = np.array([1, 0, 0, 0])
    index_finger_distal = robot.get_link("index_finger_distal")
    middle_finger_distal = robot.get_link("middle_finger_distal")
    wrist = robot.get_link("wrist")

    center = np.array([0.5, 0.5, 0.2])
    r1 = 0.1
    r2 = 0.13

    for i in range(100):
        index_finger_pos = center + np.array([np.cos(i / 90 * np.pi), np.sin(i / 90 * np.pi), 0]) * r1
        middle_finger_pos = center + np.array([np.cos(i / 90 * np.pi), np.sin(i / 90 * np.pi), 0]) * r2
        wrist_pos = index_finger_pos - np.array([0, 0, 0.20])

        target_1.set_qpos(np.concatenate([index_finger_pos, target_quat]))
        target_2.set_qpos(np.concatenate([middle_finger_pos, target_quat]))
        target_3.set_qpos(np.concatenate([wrist_pos, target_quat]))

        qpos = robot.inverse_kinematics_multilink(
            links=[index_finger_distal, middle_finger_distal, wrist],
            poss=[index_finger_pos, middle_finger_pos, wrist_pos],
        )

        robot.set_qpos(qpos)
        scene.visualizer.update(force=True)


if __name__ == "__main__":
    main()
