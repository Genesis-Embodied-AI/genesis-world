import argparse

import numpy as np

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
        ),
        viewer_options=viewer_options,
        show_viewer=args.vis,
    )

    plane = scene.add_entity(
        gs.morphs.Plane(),
    )
    franka = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
    )
    scene.build()

    joints_name = (
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
        "joint7",
        "finger_joint1",
        "finger_joint2",
    )
    motors_dof_idx = [franka.get_joint(name).dofs_idx_local[0] for name in joints_name]

    franka.set_dofs_kp(
        np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]),
        motors_dof_idx,
    )
    franka.set_dofs_kv(
        np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]),
        motors_dof_idx,
    )
    franka.set_dofs_force_range(
        np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
        np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]),
        motors_dof_idx,
    )
    # Hard reset
    for i in range(150):
        if i < 50:
            franka.set_dofs_position(np.array([1, 1, 0, 0, 0, 0, 0, 0.04, 0.04]), motors_dof_idx)
        elif i < 100:
            franka.set_dofs_position(np.array([-1, 0.8, 1, -2, 1, 0.5, -0.5, 0.04, 0.04]), motors_dof_idx)
        else:
            franka.set_dofs_position(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]), motors_dof_idx)

        scene.step()

    # PD control
    for i in range(1250):
        if i == 0:
            franka.control_dofs_position(
                np.array([1, 1, 0, 0, 0, 0, 0, 0.04, 0.04]),
                motors_dof_idx,
            )
        elif i == 250:
            franka.control_dofs_position(
                np.array([-1, 0.8, 1, -2, 1, 0.5, -0.5, 0.04, 0.04]),
                motors_dof_idx,
            )
        elif i == 500:
            franka.control_dofs_position(
                np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]),
                motors_dof_idx,
            )
        elif i == 750:
            # control first dof with velocity, and the rest with position
            franka.control_dofs_position(
                np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])[1:],
                motors_dof_idx[1:],
            )
            franka.control_dofs_velocity(
                np.array([1.0, 0, 0, 0, 0, 0, 0, 0, 0])[:1],
                motors_dof_idx[:1],
            )
        elif i == 1000:
            franka.control_dofs_force(
                np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]),
                motors_dof_idx,
            )
        # This is the internal control force computed based on the given control command
        # If using force control, it's the same as the given control command
        print("control force:", franka.get_dofs_control_force(motors_dof_idx))

        scene.step()


if __name__ == "__main__":
    main()
