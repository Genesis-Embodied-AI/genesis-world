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

    dofs_idx_local = []
    for joint in robot.joints:
        if joint.name in (
            "wrist_joint",
            "index_finger_joint1",
            "index_finger_joint2",
            "index_finger_joint3",
        ):
            dofs_idx_local += joint.dofs_idx_local

    center = np.array([0.033, -0.01, 0.42])
    r1 = 0.05

    for i in range(200):
        index_finger_pos = center + np.array([0.0, np.cos(i / 90 * np.pi) - 1.0, np.sin(i / 90 * np.pi) - 1.0]) * r1

        target_1.set_qpos(np.concatenate([index_finger_pos, target_quat]))

        qpos = robot.inverse_kinematics_multilink(
            links=[index_finger_distal],  # IK targets
            poss=[index_finger_pos],
            dofs_idx_local=dofs_idx_local,  # IK wrt these dofs
        )

        robot.set_qpos(qpos)
        scene.visualizer.update(force=True)


if __name__ == "__main__":
    main()
