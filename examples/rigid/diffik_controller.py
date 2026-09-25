import argparse

import numpy as np

import genesis as gs

config = {
    "ur5e": {"mjcf_file": "xml/universal_robots_ur5e/ur5e.xml", "end_effector_link": "ee_virtual_link"},
    "panda": {"mjcf_file": "xml/franka_emika_panda/panda.xml", "end_effector_link": "left_finger"},
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", help="Show visualization GUI")
    parser.add_argument("-g", "--gpu", action="store_true", help="Run on GPU instead of CPU")
    parser.add_argument("--robot", choices=["panda", "ur5e"], default="ur5e", help="Select robot model (panda or ur5e)")
    args = parser.parse_args()

    backend = gs.gpu if args.gpu else gs.cpu
    gs.init(backend=backend)

    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.0, -2, 1.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=40,
        ),
        show_viewer=args.vis,
        show_FPS=False,
    )

    plane = scene.add_entity(
        gs.morphs.Plane(),
        material=gs.materials.Kinematic(),
    )

    robot_config = config[args.robot]
    mjcf_file = robot_config["mjcf_file"]
    robot = scene.add_entity(
        gs.morphs.MJCF(
            file=mjcf_file,
        ),
        material=gs.materials.Kinematic(),
    )

    print("links=", robot.links)

    target_entity = scene.add_entity(
        gs.morphs.Mesh(
            file="meshes/axis.obj",
            scale=0.10,
        ),
        material=gs.materials.Kinematic(),
        surface=gs.surfaces.Default(
            color=(1, 0.5, 0.5, 1),
        ),
    )
    scene.build()

    target_quat = np.array([0, 1, 0, 0])
    center = np.array([0.5, 0, 0.5])
    r = 0.1
    damping = 1e-4
    diag = damping * np.eye(6)

    end_effector_link = robot_config["end_effector_link"]
    ee_link = robot.get_link(end_effector_link)

    for i in range(0, 2000):
        target_pos = center + np.array([np.cos(i / 360 * np.pi), np.sin(i / 360 * np.pi), 0]) * r

        target_entity.set_qpos(np.concatenate([target_pos, target_quat]))

        # Position error.
        error_pos = target_pos - ee_link.get_pos().cpu().numpy()

        # Orientation error.
        ee_quat = ee_link.get_quat().cpu().numpy()
        error_quat = gs.transform_quat_by_quat(gs.inv_quat(ee_quat), target_quat)
        error_rotvec = gs.quat_to_rotvec(error_quat)

        error = np.concatenate([error_pos, error_rotvec])

        jac = robot.get_jacobian(link=ee_link).cpu().numpy()
        dq = jac.T @ np.linalg.solve(jac @ jac.T + diag, error)
        q = robot.get_qpos().cpu().numpy() + dq

        robot.set_qpos(q)
        scene.visualizer.update(force=True)


if __name__ == "__main__":
    main()
