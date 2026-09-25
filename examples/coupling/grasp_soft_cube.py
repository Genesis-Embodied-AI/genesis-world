import argparse

import numpy as np

import genesis as gs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", help="Show visualization GUI")
    args = parser.parse_args()

    gs.init(backend=gs.cpu)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=5e-3,
            substeps=15,
        ),
        mpm_options=gs.options.MPMOptions(
            grid_density=128,
            lower_bound=(0.55, -0.1, -0.05),
            upper_bound=(0.75, 0.1, 0.3),
        ),
        vis_options=gs.options.VisOptions(
            visualize_mpm_boundary=True,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3, -1, 1.5),
            camera_lookat=(0.0, 0.0, 0.0),
            camera_fov=30,
        ),
        show_viewer=args.vis,
    )

    plane = scene.add_entity(
        gs.morphs.Plane(),
    )
    cube = scene.add_entity(
        material=gs.materials.MPM.Elastic(),
        morph=gs.morphs.Box(
            size=(0.04, 0.04, 0.04),
            pos=(0.65, 0.0, 0.025),
            euler=(0, 0, 0),
        ),
        surface=gs.surfaces.Default(
            vis_mode="particle",
        ),
    )
    franka = scene.add_entity(
        gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
        ),
        material=gs.materials.Rigid(
            coup_friction=1.0,
        ),
    )

    scene.build()

    motors_dof = np.arange(7)
    fingers_dof = np.arange(7, 9)

    franka.set_dofs_kp(
        np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]),
    )
    franka.set_dofs_kv(
        np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]),
    )
    franka.set_dofs_force_range(
        np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
        np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]),
    )

    end_effector = franka.get_link("hand")
    # move to pre-grasp pose
    qpos = franka.inverse_kinematics(
        link=end_effector,
        pos=np.array([0.64, 0.0, 0.135]),
        quat=np.array([0, 1, 0, 0]),
    )
    qpos[-2:] = 0.03
    franka.set_dofs_position(qpos[:-2], motors_dof)
    franka.set_dofs_position(qpos[-2:], fingers_dof)

    # grasp with 1N force
    franka.control_dofs_position(qpos[:-2], motors_dof)
    franka.control_dofs_force(np.array([-1, -1]), fingers_dof)

    for i in range(100):
        scene.step()

    # lift
    for i in range(300):
        qpos = franka.inverse_kinematics(
            link=end_effector,
            pos=np.array([0.64, 0.0, 0.135 + 0.0005 * i]),
            quat=np.array([0, 1, 0, 0]),
        )
        franka.control_dofs_position(qpos[:-2], motors_dof)
        scene.step()

    for i in range(100):
        scene.step()


if __name__ == "__main__":
    main()
