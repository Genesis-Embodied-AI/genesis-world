import argparse

import numpy as np

import genesis as gs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", help="Show visualization GUI")
    args = parser.parse_args()

    # Batched inverse kinematics over 64 environments is what this example times, so it needs a GPU.
    gs.init(backend=gs.gpu, precision="32", logging_level="info")
    np.set_printoptions(precision=7, suppress=True)

    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.0, -2, 1.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=40,
        ),
        show_viewer=args.vis,
    )

    plane = scene.add_entity(
        gs.morphs.Plane(),
        material=gs.materials.Kinematic(),
    )
    robot = scene.add_entity(
        gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
        ),
        material=gs.materials.Kinematic(),
    )

    target_entity = scene.add_entity(
        gs.morphs.Mesh(
            file="meshes/axis.obj",
            scale=0.15,
        ),
        material=gs.materials.Kinematic(),
        surface=gs.surfaces.Default(
            color=(1, 0.5, 0.5, 1),
        ),
    )

    n_envs = 64
    scene.build(n_envs=n_envs, env_spacing=(1.0, 1.0))

    target_quat = np.tile(np.array([0, 1, 0, 0]), [n_envs, 1])  # pointing downwards
    center = np.tile(np.array([0.4, -0.2, 0.25]), [n_envs, 1])
    angular_speed = np.random.uniform(-10, 10, n_envs)
    r = 0.1

    ee_link = robot.get_link("hand")

    for i in range(720):
        target_pos = np.zeros([n_envs, 3])
        target_pos[:, 0] = center[:, 0] + np.cos(i / 360 * np.pi * angular_speed) * r
        target_pos[:, 1] = center[:, 1] + np.sin(i / 360 * np.pi * angular_speed) * r
        target_pos[:, 2] = center[:, 2]
        target_q = np.hstack([target_pos, target_quat])

        target_entity.set_qpos(target_q)
        q = robot.inverse_kinematics(
            link=ee_link,
            pos=target_pos,
            quat=target_quat,
            rot_mask=[False, False, True],  # for demo purpose: only restrict direction of z-axis
        )

        robot.set_qpos(q)
        scene.visualizer.update(force=True)


if __name__ == "__main__":
    main()
