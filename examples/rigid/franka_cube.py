import argparse

import numpy as np
import genesis as gs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", help="Show visualization GUI")
    args = parser.parse_args()

    gs.init(backend=gs.cpu, precision="32")
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
        ),
        rigid_options=gs.options.RigidOptions(
            box_box_detection=True,
        ),
        viewer_options=gs.options.ViewerOptions(
            res=(960, 640),
            camera_pos=(3, -1, 1.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=30,
        ),
        show_viewer=args.vis,
    )

    plane = scene.add_entity(
        gs.morphs.Plane(),
    )
    franka = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
    )

    cube = scene.add_entity(
        gs.morphs.Box(
            size=(0.04, 0.04, 0.04),
            pos=(0.65, 0.0, 0.02),
        )
    )
    scene.build()

    motors_dof = np.arange(7)
    fingers_dof = np.arange(7, 9)
    franka.set_dofs_kp([100.0, 100.0], fingers_dof)
    franka.set_dofs_kv([10.0, 10.0], fingers_dof)
    qpos = np.array([-1.0124, 1.5559, 1.3662, -1.6878, -1.5799, 1.7757, 1.4602, 0.04, 0.04])
    franka.set_qpos(qpos)
    scene.step()

    end_effector = franka.get_link("hand")
    qpos = franka.inverse_kinematics(
        link=end_effector,
        pos=np.array([0.65, 0.0, 0.135]),
        quat=np.array([0, 1, 0, 0]),
    )

    franka.control_dofs_position(qpos[:-2], motors_dof)

    # hold
    for i in range(100):
        print("hold", i)
        scene.step()

    # grasp
    finder_pos = -0.0
    for i in range(100):
        print("grasp", i)
        franka.control_dofs_position(qpos[:-2], motors_dof)
        franka.control_dofs_position(np.array([finder_pos, finder_pos]), fingers_dof)
        scene.step()

    # lift
    qpos = franka.inverse_kinematics(
        link=end_effector,
        pos=np.array([0.65, 0.0, 0.3]),
        quat=np.array([0, 1, 0, 0]),
    )
    for i in range(200):
        print("lift", i)
        franka.control_dofs_position(qpos[:-2], motors_dof)
        franka.control_dofs_position(np.array([finder_pos, finder_pos]), fingers_dof)
        scene.step()


if __name__ == "__main__":
    main()
