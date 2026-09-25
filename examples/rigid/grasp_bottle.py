import argparse

import numpy as np

import genesis as gs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", help="Show visualization GUI")
    parser.add_argument("-b", "--num-envs", type=int, default=49, help="Number of parallel environments")
    args = parser.parse_args()

    # The 49 parallel grasps are the point of this example, so it needs a GPU.
    gs.init(backend=gs.gpu)

    viewer_options = gs.options.ViewerOptions(
        camera_pos=(3, -1, 1.5),
        camera_lookat=(0.0, 0.0, 0.0),
        camera_fov=30,
    )

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
        ),
        viewer_options=viewer_options,
        show_viewer=args.vis,
    )

    plane = scene.add_entity(
        gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True),
    )
    bottle = scene.add_entity(
        material=gs.materials.Rigid(
            rho=300,
        ),
        morph=gs.morphs.URDF(
            file="urdf/3763/mobility_vhacd.urdf",
            scale=0.09,
            pos=(0.65, 0.0, 0.036),
            euler=(0, 90, 0),
        ),
    )
    franka = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
    )

    scene.build(n_envs=args.num_envs, env_spacing=(1, 1))

    motors_dof = np.arange(7)
    fingers_dof = np.arange(7, 9)

    if args.num_envs == 0:
        franka.set_qpos(np.array([1.56, -0.72, -0.02, -2.09, 0.04, 1.33, 2.4, 0.01, 0.01]))
    else:
        franka.set_qpos(np.array([[1.56, -0.72, -0.02, -2.09, 0.04, 1.33, 2.4, 0.01, 0.01]] * args.num_envs))
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
        pos=np.array([0.65, 0.0, 0.25]) if args.num_envs == 0 else np.array([[0.65, 0.0, 0.25]] * args.num_envs),
        quat=np.array([0, 1, 0, 0]) if args.num_envs == 0 else np.array([[0, 1, 0, 0]] * args.num_envs),
    )
    qpos[..., -2:] = 0.04

    path = franka.plan_path(qpos)
    for waypoint in path:
        franka.control_dofs_position(waypoint)
        scene.step()
    for i in range(30):
        scene.step()

    # reach
    qpos = franka.inverse_kinematics(
        link=end_effector,
        pos=np.array([0.65, 0.0, 0.142]) if args.num_envs == 0 else np.array([[0.65, 0.0, 0.142]] * args.num_envs),
        quat=np.array([0, 1, 0, 0]) if args.num_envs == 0 else np.array([[0, 1, 0, 0]] * args.num_envs),
    )
    franka.control_dofs_position(qpos[..., :-2], motors_dof)
    for i in range(100):
        scene.step()

    # grasp
    franka.control_dofs_position(qpos[..., :-2], motors_dof)
    franka.control_dofs_position(
        np.array([0, 0]) if args.num_envs == 0 else np.array([[0, 0]] * args.num_envs), fingers_dof
    )  # you can use position control
    for i in range(100):
        scene.step()

    # lift
    qpos = franka.inverse_kinematics(
        link=end_effector,
        pos=np.array([0.65, 0.0, 0.3]) if args.num_envs == 0 else np.array([[0.65, 0.0, 0.3]] * args.num_envs),
        quat=np.array([0, 1, 0, 0]) if args.num_envs == 0 else np.array([[0, 1, 0, 0]] * args.num_envs),
    )
    franka.control_dofs_position(qpos[..., :-2], motors_dof)
    franka.control_dofs_force(
        np.array([-20, -20]) if args.num_envs == 0 else np.array([[-20, -20]] * args.num_envs), fingers_dof
    )  # can also use force control
    for i in range(1000):
        scene.step()


if __name__ == "__main__":
    main()
